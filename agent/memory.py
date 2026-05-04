"""
agent/memory.py — Stateful Conversation History
===============================================

This module handles short-term memory for the conversational agent.
Without memory, the LLM treats every question as the very first question.

Key responsibilities:
1. Store and retrieve conversation history from Redis.
2. Enforce a Sliding Window (truncate old messages) to prevent exceeding
   the LLM's context limit or confusing it with stale topics.
3. Automatically expire idle sessions (TTL) so we don't leak memory.

Design decisions:
-----------------
* Storage: Redis (In-Memory Data Store)
  Why: It's lightning fast, naturally supports TTL (auto-expiration), and
  allows us to run multiple agents in parallel (stateless backend).
* Truncation: Sliding Window
  We keep only the `MAX_HISTORY_MESSAGES` most recent messages.
  (In a V2, you would summarise the dropped messages instead of deleting them).
"""

import json
from typing import List, Dict

import redis

from google import genai

from core.metadata_store import DocumentMetadata
from config import REDIS_URL, MAX_HISTORY_MESSAGES, REDIS_SESSION_TTL, GEMINI_API_KEY, GENERATION_MODEL
from utils.logger import get_logger

log = get_logger(__name__)


class SessionMemory:
    """
    Manages conversational memory for a specific user session using Redis.
    """

    def __init__(self, session_id: str) -> None:
        """
        Args:
            session_id: A unique identifier for the user's session (e.g., a UUID).
        """
        self.session_id = session_id
        # The key we'll use to store this session's data in Redis
        self.redis_key = f"chat_history:{self.session_id}"
        self.metadata_key = f"metadata:{self.session_id}"
        
        # Connect to Redis. decode_responses=True means we get strings back instead of bytes
        self.client = redis.from_url(REDIS_URL, decode_responses=True)
        log.debug(f"Initialised memory for session: {self.session_id}")

    def save_metadata(self, metadata: DocumentMetadata) -> None:
        try:
            from dataclasses import asdict
            data = asdict(metadata)
            self.client.set(self.metadata_key, json.dumps(data), ex=REDIS_SESSION_TTL)
            log.info(f"Saved document metadata for session {self.session_id}")
        except redis.RedisError as e:
            log.error(f"Redis write failed for metadata in session {self.session_id}: {str(e)}")

    def get_metadata(self) -> DocumentMetadata | None:
        try:
            raw_data = self.client.get(self.metadata_key)
            if not raw_data:
                return None
            data = json.loads(raw_data)
            return DocumentMetadata(**data)
        except redis.RedisError as e:
            log.error(f"Redis read failed for metadata in session {self.session_id}: {str(e)}")
            return None

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retrieve the conversation history from Redis.
        
        Returns:
            A list of message dictionaries: [{"role": "user", "content": "..."}, ...]
        """
        try:
            raw_data = self.client.get(self.redis_key)
            if not raw_data:
                return []
                
            history = json.loads(raw_data)
            return history
            
        except redis.RedisError as e:
            log.error(f"Redis read failed for session {self.session_id}: {str(e)}")
            # Fail gracefully: return empty history rather than crashing the chat
            return []

    def save_turn(self, user_message: str, assistant_message: str) -> None:
        """
        Save a full conversational turn (User asks -> Assistant answers).
        Enforces the sliding window limit and resets the TTL expiration.
        """
        try:
            # 1. Load existing history
            history = self.get_history()
            
            # 2. Append new messages
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_message})
            
            # 3. Truncate to sliding window limit (MAX_HISTORY_MESSAGES) and summarize overflow
            if len(history) > MAX_HISTORY_MESSAGES:
                overflow_messages = history[:-MAX_HISTORY_MESSAGES]
                recent_messages = history[-MAX_HISTORY_MESSAGES:]
                
                text_to_summarize = ""
                for msg in overflow_messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    text_to_summarize += f"{role}: {msg['content']}\n"
                    
                summary = self._summarize(text_to_summarize)
                history = [{"role": "assistant", "content": f"[SYSTEM_SUMMARY: {summary}]"}] + recent_messages
                log.debug(f"Summarised older history for session {self.session_id}.")
            
            # 4. Save back to Redis with a TTL (Time To Live)
            # ex=REDIS_SESSION_TTL means "delete this key automatically if it's not updated for X seconds"
            self.client.set(
                self.redis_key, 
                json.dumps(history), 
                ex=REDIS_SESSION_TTL
            )
            log.info(f"Saved turn for session {self.session_id}. History size: {len(history)} messages.")
            
        except redis.RedisError as e:
            log.error(f"Redis write failed for session {self.session_id}: {str(e)}")

    def clear(self) -> None:
        """
        Wipe the conversation history for this session.
        Useful for a "New Chat" button in the UI.
        """
        try:
            self.client.delete(self.redis_key)
            log.info(f"Cleared memory for session: {self.session_id}")
        except redis.RedisError as e:
            log.error(f"Redis delete failed for session {self.session_id}: {str(e)}")

    def _summarize(self, text_to_summarize: str) -> str:
        """Use Gemini to summarize the overflow conversation history."""
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"Summarize the following conversation history concisely in 1-2 sentences. Focus on the main topics discussed:\n\n{text_to_summarize}"
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            log.error(f"Failed to summarize history: {e}")
            return "Previous conversation summary unavailable."

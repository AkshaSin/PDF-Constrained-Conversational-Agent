"""
agent/generator.py — Constrained LLM Generation
===============================================

This module takes the retrieved document chunks and the conversation history,
and feeds them to the Google Gemini LLM to generate the final answer.

Key responsibilities:
1. Anti-Hallucination Layer 2: The Constrained Prompt.
   We explicitly command the LLM to refuse to answer if the context does not
   contain the information. 
2. Prompt Engineering.
   We inject the retrieved chunks into the prompt as XML-like tags (<context>)
   to help the LLM clearly separate the user's question from the source material.
3. Chat History injection.
   We format the Redis history into the prompt so the LLM has context for
   follow-up questions.
"""

from typing import List, Dict, Generator

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from config import GENERATION_MODEL
from core.chunker import Chunk
from utils.logger import get_logger

log = get_logger(__name__)


# The SYSTEM PROMPT is the most critical part of an Anti-Hallucination system.
# It defines the strict rules the LLM must follow.
SYSTEM_PROMPT = """You are a precise, professional AI assistant tasked with answering questions based ONLY on the provided document excerpts.

CRITICAL RULES:
1. You may ONLY use the information provided in the <context> tags below.
2. If the answer cannot be found in the <context>, you MUST state exactly: "I cannot answer that based on the provided document." Do not attempt to guess or use outside knowledge.
3. Do not mention that you are reading from "context tags" or "chunks" — simply answer the question naturally.
4. If the user is making small talk (e.g., "hello", "how are you"), you may answer politely, but gently guide them back to asking about the document.

<context>
{context_text}
</context>"""


class LLMGenerator:
    """
    Handles the generation of constrained answers using Google Gemini.
    """

    def __init__(self) -> None:
        """Initialise the generation model."""
        log.info(f"Initialising generator with model: {GENERATION_MODEL}")
        # We assume genai.configure() was already called by the Embedder,
        # but the API handles redundant calls safely.
        
        # Configure generation parameters to minimise hallucination
        # temperature=0.1 means the model will be highly deterministic and
        # less likely to "creatively" invent facts.
        self.config = GenerationConfig(
            temperature=0.1,
            top_p=0.8,
        )
        
        # Instantiate the model. We use system_instruction to enforce the core rules.
        # (Note: we will update the system_instruction dynamically per request
        # because the {context_text} changes for every question).
        self.model = genai.GenerativeModel(GENERATION_MODEL)
        log.debug("Generator initialised successfully.")

    def _format_context(self, chunks: List[Chunk]) -> str:
        """
        Format the retrieved chunks into a single string for the prompt.
        """
        if not chunks:
            return "No relevant context found in the document."
            
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            # We add a source ID so the LLM can differentiate distinct passages
            formatted_chunks.append(f"--- Excerpt {i} ---\n{chunk.text}\n")
            
        return "\n".join(formatted_chunks)

    def generate_stream(
        self, 
        query: str, 
        chunks: List[Chunk], 
        history: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        """
        Generate an answer and yield it chunk-by-chunk (streaming).
        
        Args:
            query: The user's specific question.
            chunks: The Top-K chunks retrieved from the reranker.
            history: The recent conversation history from Redis.
            
        Yields:
            Strings representing the partial LLM response as it is generated.
        """
        context_text = self._format_context(chunks)
        system_instruction = SYSTEM_PROMPT.format(context_text=context_text)
        
        # Build the conversation manually to ensure the system prompt and history
        # are perfectly structured for the generation call, regardless of SDK version.
        prompt_parts = [system_instruction, "\n\n--- CONVERSATION HISTORY ---"]
        
        if not history:
            prompt_parts.append("\n(No previous history)")
        else:
            for msg in history:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                prompt_parts.append(f"\n{role_label}: {msg['content']}")
                
        prompt_parts.append(f"\n\n--- CURRENT QUESTION ---\nUser: {query}\nAssistant: ")
        
        final_prompt = "".join(prompt_parts)
        log.debug("Sending fully constructed prompt to Gemini...")
        
        try:
            # Generate response as a stream
            response = self.model.generate_content(
                final_prompt,
                generation_config=self.config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            log.error(f"Generation failed: {str(e)}")
            yield f"An error occurred while contacting the LLM: {str(e)}"

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

from typing import List, Dict, Generator, Any

from google import genai
from google.genai import types

from config import GENERATION_MODEL, GEMINI_API_KEY
from core.chunker import Chunk
from utils.logger import get_logger

log = get_logger(__name__)


# The SYSTEM PROMPT is the most critical part of an Anti-Hallucination system.
# It defines the strict rules the LLM must follow.
SYSTEM_PROMPT = """You are a precise, professional AI assistant tasked with answering questions based ONLY on the provided document excerpts or verified facts.

CRITICAL RULES:
1. You may ONLY use the information provided in the <context> or <verified_fact> tags below.
2. If the answer cannot be found in the provided tags, you MUST state exactly: "I cannot answer that based on the provided document." Do not attempt to guess or use outside knowledge.
3. Do not mention that you are reading from "context tags", "verified facts", or "chunks" — simply answer the question naturally.
4. If the user is making small talk (e.g., "hello", "how are you"), you may answer politely, but gently guide them back to asking about the document.
5. When using information from an excerpt, you MUST append a citation to the end of your answer (or end of the relevant sentence), formatted exactly as `[Source: Page X]`, based on the "Page" label provided in the excerpt header. If using a verified fact, you do not need to add a page citation unless it's explicitly mentioned in the fact.
6. EXTREMELY IMPORTANT: If you are asked to provide information that is clearly not present in the excerpts, you MUST refuse. Hallucinating facts, figures, or details that are not in the provided context is STRICTLY FORBIDDEN.

<tools>
You have access to Python-executed tools to compute exact statistics.
Available tools:
  - count_word
    Description: Count exact occurrences of a word in the document.
               Note: Only matches standalone words, not phrases or sub-words.
    Parameters: {{"word": "the exact word to count"}}
  - get_page_count
    Description: Get the total number of pages in the document.
    Parameters: {{}}
  - get_page_content
    Description: Retrieve the exact full text of a specific page.
    Parameters: {{"page_num": "The 1-indexed page number"}}
  - find_all_occurrences
    Description: Find all occurrences of a specific phrase and return their page numbers and context snippets.
    Parameters: {{"phrase": "The exact phrase to search for"}}
</tools>

RESPONSE FORMAT RULES (strict, no exceptions):

1. If you need to call a tool, your ENTIRE response must be ONLY:
   <tool_call>{{"name": "...", "args": {{...}}}}</tool_call>
   No preamble. No thinking block. No other text whatsoever.

2. If you are providing the FINAL answer (no more tools needed), your response MUST follow this exact structure:
   <thinking>
   [Your reasoning: what you found in the context, how you are interpreting it, and how you arrived at the answer]
   </thinking>
   [Your final answer to the user, including citations like [Source: Page X]]

3. The two modes are mutually exclusive:
   - Tool calls NEVER contain a <thinking> block.
   - Final answers ALWAYS contain a <thinking> block — no skipping, even for short answers.

4. Citation placement: [Source: Page X] markers belong ONLY in the final answer text, NEVER inside the <thinking> block.

<context>
{context_text}
</context>

<verified_fact>
{verified_fact_text}
</verified_fact>"""


class LLMGenerator:
    """
    Handles the generation of constrained answers using Google Gemini.
    """

    def __init__(self) -> None:
        """Initialise the generation model."""
        log.info(f"Initialising generator with model: {GENERATION_MODEL}")
        self._api_key = GEMINI_API_KEY
        
        # Configure generation parameters to minimise hallucination.
        # thinking_budget=0 disables Gemini 2.5-flash's native thinking tokens,
        # which are billed but invisible — they can add hundreds of tokens per
        # response without contributing to the visible output.
        self.config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.8,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        log.debug("Generator initialised successfully.")

    def _format_context(self, chunks: List[Chunk]) -> str:
        """Format the retrieved chunks into a single string for the prompt."""
        if not chunks:
            return "No relevant context found in the document."
            
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            formatted_chunks.append(f"--- Excerpt {i} (Page {chunk.page_number}) ---\n{chunk.text}\n")
            
        return "\n".join(formatted_chunks)

    def generate_stream(
        self, 
        query: str, 
        chunks: List[Chunk], 
        history: List[Dict[str, str]],
        verified_fact: str = None
    ) -> Generator[str, None, None]:
        """Generate an answer and yield it chunk-by-chunk (streaming) without tools."""
        context_text = self._format_context(chunks) if chunks else "No relevant context found in the document."
        fact_text = verified_fact if verified_fact else "None"
        
        system_instruction = SYSTEM_PROMPT.replace("{context_text}", context_text).replace("{verified_fact_text}", fact_text)
        prompt_parts = [system_instruction, "\n\n--- CONVERSATION HISTORY ---"]
        
        if not history:
            prompt_parts.append("\n(No previous history)")
        else:
            for msg in history:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                prompt_parts.append(f"\n{role_label}: {msg['content']}")
                
        prompt_parts.append(f"\n\n--- CURRENT QUESTION ---\nUser: {query}\nAssistant: ")
        final_prompt = "".join(prompt_parts)
        
        try:
            client = genai.Client(api_key=self._api_key)
            response_stream = client.models.generate_content_stream(
                model=GENERATION_MODEL,
                contents=final_prompt,
                config=self.config
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            log.error(f"Generation failed: {str(e)}")
            yield f"An error occurred while contacting the LLM: {str(e)}"

    def generate_with_tools(
        self, 
        query: str, 
        chunks: List[Chunk], 
        history: List[Dict[str, str]],
        tool_executor,
        verified_fact: str = None,
        max_iterations: int = 3
    ) -> Generator[Any, None, None]:
        """
        Agentic loop: yields tool statuses or final streamed chunks.
        """
        import json
        import re
        
        context_text = self._format_context(chunks) if chunks else "No relevant context found in the document."
        fact_text = verified_fact if verified_fact else "None"
        
        system_instruction = SYSTEM_PROMPT.replace("{context_text}", context_text).replace("{verified_fact_text}", fact_text)
        prompt_parts = [system_instruction, "\n\n--- CONVERSATION HISTORY ---"]
        
        if not history:
            prompt_parts.append("\n(No previous history)")
        else:
            for msg in history:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                prompt_parts.append(f"\n{role_label}: {msg['content']}")
                
        prompt_parts.append(f"\n\n--- CURRENT QUESTION ---\nUser: {query}\nAssistant: ")
        
        client = genai.Client(api_key=self._api_key)
        
        for iteration in range(max_iterations):
            final_prompt = "".join(prompt_parts)
            log.debug(f"Tool iteration {iteration+1}/{max_iterations}")
            
            try:
                response_stream = client.models.generate_content_stream(
                    model=GENERATION_MODEL,
                    contents=final_prompt,
                    config=self.config
                )
                
                buffer = ""
                is_tool_call = False
                stream_buffer_active = True

                for chunk in response_stream:
                    if not chunk.text:
                        continue

                    if stream_buffer_active:
                        buffer += chunk.text
                        stripped = buffer.lstrip()

                        if stripped.startswith("<tool_call"):
                            is_tool_call = True
                            stream_buffer_active = False
                        elif stripped.startswith("<thinking"):
                            # Model emitted a <thinking> block before (possibly) a tool call.
                            # Keep buffering silently until </thinking> is complete, then
                            # decide based on what follows it.
                            if "</thinking>" in buffer:
                                after_thinking = buffer.split("</thinking>", 1)[1].lstrip()
                                if after_thinking.startswith("<tool_call"):
                                    is_tool_call = True
                                    stream_buffer_active = False
                                elif len(after_thinking) > 30 and "<tool_call" not in after_thinking:
                                    # Thinking block done, followed by regular answer text — flush
                                    yield buffer
                                    buffer = ""
                                    stream_buffer_active = False
                                    is_tool_call = False
                                # else: </thinking> just arrived, not enough after-text yet — keep buffering
                            # else: still inside the thinking block — keep buffering
                        elif len(stripped) > 50 and "<tool_call" not in buffer:
                            # No thinking block, no tool call incoming — flush as regular response
                            yield buffer
                            buffer = ""
                            stream_buffer_active = False
                            is_tool_call = False
                    else:
                        if is_tool_call:
                            buffer += chunk.text
                        else:
                            yield chunk.text
                
                if is_tool_call or "<tool_call>" in buffer:
                    # Execute tool
                    match = re.search(r"<tool_call>(.*?)</tool_call>", buffer, re.DOTALL)
                    if match:
                        try:
                            call = json.loads(match.group(1))
                            tool_name = call.get("name")
                            args = call.get("args", {})
                            yield {"type": "tool_status", "name": tool_name}
                            
                            result = tool_executor.execute(tool_name, **args)
                            res_str = result.error if result.error else str(result.output)
                            
                            prompt_parts.append(f"{buffer}\nUser: <tool_result name='{tool_name}'>{res_str}</tool_result>\nAssistant: ")
                        except Exception as e:
                            prompt_parts.append(f"{buffer}\nUser: <tool_result name='error'>Failed to parse or execute: {str(e)}</tool_result>\nAssistant: ")
                    else:
                        prompt_parts.append(f"{buffer}\nUser: <tool_result name='error'>Malformed tool call block.</tool_result>\nAssistant: ")
                else:
                    # Normal response that happened to be very short and buffered entirely
                    if buffer:
                        yield buffer
                    return
                    
            except Exception as e:
                log.error(f"Generation failed during tool loop: {str(e)}")
                yield f"An error occurred while contacting the LLM: {str(e)}"
                return
                
        # If we exhausted iterations
        yield "\n\n[System Error: Agent stopped after reaching maximum tool iterations to prevent runaway execution.]"

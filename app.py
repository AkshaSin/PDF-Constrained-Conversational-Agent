"""
app.py — The Main Application and UI
====================================

This is the entry point for the PDF-Constrained Conversational Agent.
It wires together all the backend modules into a Gradio web interface.

Key responsibilities:
1. Provide a UI for uploading PDFs and chatting.
2. Orchestrate the Ingestion Pipeline (Upload -> Parse -> Chunk -> Index).
3. Orchestrate the Generation Pipeline (Query -> Retrieve -> Rerank -> LLM -> UI).
4. Manage session state (linking the browser tab to a Redis memory session).
"""

import os
import uuid
import time
from typing import Dict, List, Tuple

import gradio as gr

from core.parser import PDFParser
from core.chunker import TextChunker
from core.embedder import FAISSEmbedder
from core.bm25_retriever import BM25Retriever
from core.hybrid_retriever import HybridRetriever
from config import INDEX_DIR
from core.reranker import CrossEncoderReranker
from agent.memory import SessionMemory
from agent.generator import LLMGenerator
from core.query_router import QueryRouter
from core.factual_answerer import FactualAnswerer
from core.metadata_store import DocumentMetadata
from core.tools import ToolExecutor
from collections import Counter
from utils.logger import get_logger

log = get_logger(__name__)


# =====================================================================
# UI Transform Helpers
# =====================================================================

def transform_thinking_tags(text: str) -> str:
    """
    Converts raw <thinking>...</thinking> tags in streamed LLM output into
    a Gradio-renderable HTML <details> accordion. This allows the user to
    watch the agent's reasoning stream live and collapse it afterwards.

    Safe for partial/streaming text: if a tag is only partially received
    (e.g. the stream ends mid-tag like "<thinki" or "</thinking"),
    the function returns the text unchanged to prevent broken HTML.
    """
    # Guard: closing tag partially streamed — check ALL partial prefixes
    # (e.g. "</thinki", "</thinking", etc.) before any replacement runs.
    _CLOSING = "</thinking>"
    if any(_CLOSING[:i] in text for i in range(2, len(_CLOSING))) and _CLOSING not in text:
        return text
    # Guard: opening tag partially streamed (e.g. "<thinki" or "<thinking" without ">")
    if "<thinking" in text and "<thinking>" not in text:
        return text
    text = text.replace(
        "<thinking>",
        "<details open>\n<summary>Agent Thinking...</summary>\n\n"
    )
    text = text.replace("</thinking>", "\n</details>\n\n")
    return text


def strip_thinking(text: str) -> str:
    """
    Strips <thinking>...</thinking> blocks from the final LLM response
    before saving to Redis conversation memory. This prevents the model
    from seeing its own reasoning in future turns, which bloats the
    context window and can confuse subsequent responses.
    """
    import re
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()

# =====================================================================
# Global Backend Components (Initialised once on startup)
# =====================================================================

log.info("Starting up backend components...")
embedder = FAISSEmbedder()
bm25 = BM25Retriever()
hybrid_retriever = HybridRetriever(embedder=embedder, bm25=bm25)
reranker = CrossEncoderReranker()
generator = LLMGenerator()

# Prepare persistence paths
os.makedirs(INDEX_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
FAISS_MAP_PATH = os.path.join(INDEX_DIR, "faiss_map.pkl")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25.pkl")

# Attempt to load existing indices
if embedder.load_index(FAISS_INDEX_PATH, FAISS_MAP_PATH):
    log.info("Persistent FAISS index loaded successfully.")
    
if bm25.load_index(BM25_INDEX_PATH):
    log.info("Persistent BM25 index loaded successfully.")

log.info("All backend components initialised successfully.")


# =====================================================================
# Pipeline Functions
# =====================================================================

def process_pdf(pdf_filepath: str, session_id: str) -> str:
    """
    The Ingestion Pipeline.
    Runs when a user uploads a file.
    """
    if not pdf_filepath:
        return "Please upload a PDF file first."
        
    try:
        log.info(f"Processing uploaded PDF: {pdf_filepath}")
        
        # 1. Parse (extract text, filter headers)
        with open(pdf_filepath, "rb") as f:
            raw_bytes = f.read()
        parser = PDFParser(raw_bytes)
        clean_pages, page_boundaries = parser.parse()
        
        # Calculate Metadata
        total_chars = sum(len(text) for _, text in clean_pages)
        words = []
        for _, text in clean_pages:
            words.extend(text.lower().split())
        word_count = len(words)
        word_frequencies = dict(Counter(words))
        page_count = parser.num_pages
        full_text = " ".join([text for _, text in clean_pages])
        
        metadata = DocumentMetadata(page_count, word_count, total_chars, word_frequencies, page_boundaries, full_text)
        SessionMemory(session_id).save_metadata(metadata)
        
        # 2. Chunk
        chunker = TextChunker(clean_pages)
        chunks = chunker.chunk()
        
        if not chunks:
            return "Failed to extract any text chunks from the PDF. It may be empty or an image-only PDF."
            
        # 3. Index (Build FAISS and BM25)
        start_time = time.time()
        embedder.build_index(chunks)
        bm25.build_index(chunks)
        
        # 4. Persist indices to disk
        embedder.save_index(FAISS_INDEX_PATH, FAISS_MAP_PATH)
        bm25.save_index(BM25_INDEX_PATH)
        
        end_time = time.time()
        
        msg = f"[SUCCESS] PDF processed successfully!\n- Extracted {len(chunks)} chunks.\n- Indexed in {end_time - start_time:.2f} seconds.\n\nYou can now ask questions."
        log.info(msg.replace("\n", " "))
        return msg
        
    except Exception as e:
        err_msg = f"[ERROR] Error processing PDF: {str(e)}"
        log.error(err_msg)
        return err_msg


def chat_inference(user_message, chat_history, session_id):
    """
    The Generation Pipeline.
    Runs every time the user sends a message.
    
    Gradio expects a generator that yields the full updated chat_history
    so it can stream the response to the UI.
    """
    # Initialize memory for this specific user session
    memory = SessionMemory(session_id)

    # GUARD: Check that a PDF has actually been uploaded and indexed.
    # If the user sends a message with no PDF loaded, embedder.index is None.
    # Without this check, the pipeline would run on an empty FAISS index,
    # get zero results, and the LLM might answer from general world knowledge —
    # breaking our core "PDF-constrained" guarantee.
    if embedder.index is None or embedder.index.ntotal == 0:
        chat_history.append({"role": "assistant", "content": "⚠️ System: Please upload and process a PDF document first before asking questions."})
        yield chat_history
        return
    
    # 1. Update UI immediately with the user's message and a blank assistant response
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})
    yield chat_history
    
    # 2. Retrieve Conversation Context from Redis
    recent_history = memory.get_history()
    
    # Retrieve Metadata
    metadata = memory.get_metadata()
    if not metadata:
        chat_history.append({"role": "assistant", "content": "⚠️ System: Document metadata not found. Please re-upload the PDF to initialize the semantic engine."})
        yield chat_history
        return
        
    # 3. Route Query
    router = QueryRouter()
    routing_result = router.classify(user_message)
    intent = routing_result["intent"]
    sub_intent = routing_result.get("sub_intent")
    
    verified_fact = None
    if intent == "factual":
        answerer = FactualAnswerer(metadata)
        verified_fact = answerer.answer(sub_intent, user_message)
        log.info(f"Query routed as FACTUAL. Fact: {verified_fact}")
        best_chunks = []
    elif intent == "aggregation":
        log.info("Query routed as AGGREGATION. Falling back to hybrid retrieval due to pipeline limitations.")
        candidate_chunks = hybrid_retriever.retrieve(user_message)
        best_chunks = reranker.rerank(user_message, candidate_chunks)
    else:
        log.info("Query routed as SEMANTIC.")
        candidate_chunks = hybrid_retriever.retrieve(user_message)
        best_chunks = reranker.rerank(user_message, candidate_chunks)
    
    # 5. Generation Stage (LLM)
    tool_executor = ToolExecutor(metadata)
    partial_response = ""
    # We pass the question, the retrieved context, and the short-term memory
    response_stream = generator.generate_with_tools(
        query=user_message,
        chunks=best_chunks,
        history=recent_history,
        tool_executor=tool_executor,
        verified_fact=verified_fact
    )

    first_chunk_received = False

    # Expected on-screen sequence for a tool-using query:
    # Step 1: User sends message
    # Step 2: "🔧 Running tool: ..." appears (existing tool status)
    # Step 3: Tool status is REPLACED (not appended) by empty string before final answer
    # Step 4: 🧠 Agent Thinking accordion appears, expanded, streaming reasoning
    # Step 5: </thinking> closes the accordion section
    # Step 6: Final answer text streams below the accordion
    # Step 7: Stream ends — user can click to collapse the accordion
    for chunk in response_stream:
        if isinstance(chunk, dict) and chunk.get("type") == "tool_status":
            chat_history[-1]["content"] = f"Running tool: {chunk['name']}..."
            yield chat_history
        else:
            if not first_chunk_received:
                chat_history[-1]["content"] = ""  # Explicitly clear "Running tool" preamble
                first_chunk_received = True

            partial_response += chunk
            # Only transform a display copy — keep partial_response raw for memory storage
            chat_history[-1]["content"] = transform_thinking_tags(partial_response)
            yield chat_history

    # 6. Save Turn to Memory (Post-Generation)
    # Strip <thinking> blocks before persisting so the LLM does not see its
    # own reasoning in future turns (reduces context bloat and confusion).
    memory.save_turn(user_message, strip_thinking(partial_response))


# =====================================================================
# Gradio UI Layout
# =====================================================================

with gr.Blocks(title="PDF-Constrained Agent", theme=gr.themes.Soft()) as demo:
    # A hidden state variable to hold the unique Session ID for this browser tab
    session_id_state = gr.State(lambda: str(uuid.uuid4()))
    
    gr.Markdown("# 📄 PDF-Constrained Conversational Agent")
    gr.Markdown(
        "Upload a PDF, wait for the processing to finish, and ask questions. "
        "The agent is strictly constrained to answer **only** based on the document."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Left panel: Upload and controls
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            status_output = gr.Textbox(label="System Status", value="Waiting for PDF upload...", interactive=False)
            clear_btn = gr.Button("Clear Conversation")
            
        with gr.Column(scale=3):
            # Right panel: Chat interface
            # Gradio 5.33.0 requires type="messages" explicitly to accept
            # OpenAI-style dicts. Without it, the chatbot defaults to tuples format.
            chatbot = gr.Chatbot(label="Conversation", height=600, type="messages")
            msg_input = gr.Textbox(
                label="Ask a question about the document",
                placeholder="Type your question and press Enter...",
                interactive=True
            )
            
    # --- Event Binding ---
    
    # When a PDF is uploaded, run process_pdf and update the status box
    pdf_input.upload(
        fn=process_pdf,
        inputs=[pdf_input, session_id_state],
        outputs=[status_output]
    )
    
    # When the user presses Enter in the message box:
    # 1. Run chat_inference
    # 2. Update the chatbot UI
    # 3. Clear the message input box automatically
    msg_input.submit(
        fn=chat_inference,
        inputs=[msg_input, chatbot, session_id_state],
        outputs=[chatbot]
    ).then(
        fn=lambda: "", # Clear the input box
        inputs=None,
        outputs=[msg_input]
    )
    
    # Clear conversation
    def clear_memory(session_id):
        SessionMemory(session_id).clear()
        return [], ""
        
    clear_btn.click(
        fn=clear_memory,
        inputs=[session_id_state],
        outputs=[chatbot, msg_input]
    )


# Standard Python execution guard
if __name__ == "__main__":
    log.info("Starting Gradio server...")
    # launch() starts the local web server.
    # share=False keeps it local. Set True to generate a public link.
    demo.launch(server_name="0.0.0.0", share=False)

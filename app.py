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
from typing import Tuple, List

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
from utils.logger import get_logger

log = get_logger(__name__)

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

def process_pdf(pdf_filepath: str) -> str:
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
        clean_text = parser.parse()
        
        # 2. Chunk
        chunker = TextChunker(clean_text)
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
        
        msg = f"✅ PDF processed successfully!\n- Extracted {len(chunks)} chunks.\n- Indexed in {end_time - start_time:.2f} seconds.\n\nYou can now ask questions."
        log.info(msg.replace("\n", " "))
        return msg
        
    except Exception as e:
        err_msg = f"❌ Error processing PDF: {str(e)}"
        log.error(err_msg)
        return err_msg


def chat_inference(
    user_message: str, 
    chat_history: List[dict], 
    session_id: str
):
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
        yield [{"role": "assistant", "content": "⚠️ Please upload and process a PDF document first before asking questions."}]
        return
    
    # 1. Update UI immediately with the user's message and a blank assistant response
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})
    yield chat_history
    
    # 2. Retrieve Conversation Context from Redis
    recent_history = memory.get_history()
    
    # 3. Retrieval Stage (Hybrid: FAISS + BM25)
    candidate_chunks = hybrid_retriever.retrieve(user_message)
    
    # 4. Reranking Stage (Cross-Encoder)
    best_chunks = reranker.rerank(user_message, candidate_chunks)
    
    # 5. Generation Stage (LLM)
    partial_response = ""
    # We pass the question, the retrieved context, and the short-term memory
    response_stream = generator.generate_stream(
        query=user_message,
        chunks=best_chunks,
        history=recent_history
    )
    
    for chunk_text in response_stream:
        partial_response += chunk_text
        # Update the blank assistant response with the new chunk
        chat_history[-1]["content"] = partial_response
        yield chat_history
        
    # 6. Save Turn to Memory (Post-Generation)
    # Once the stream finishes, we save the full exchange to Redis
    memory.save_turn(user_message, partial_response)


# =====================================================================
# Gradio UI Layout
# =====================================================================

with gr.Blocks(title="PDF-Constrained Agent") as demo:
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
            # Gradio 6+ removed the `type` parameter entirely — messages dict format
            # {"role": "user"/"assistant", "content": "..."} is now the only format.
            chatbot = gr.Chatbot(label="Conversation", height=600)
            msg_input = gr.Textbox(
                label="Ask a question about the document",
                placeholder="Type your question and press Enter...",
                interactive=True
            )
            
    # --- Event Binding ---
    
    # When a PDF is uploaded, run process_pdf and update the status box
    pdf_input.upload(
        fn=process_pdf,
        inputs=[pdf_input],
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
    # theme is passed here as per Gradio 6.0+ API.
    # share=False keeps it local. Set True to generate a public link.
    demo.launch(server_name="127.0.0.1", share=False, theme=gr.themes.Soft())

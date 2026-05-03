"""
core/reranker.py — Cross-Encoder Reranking Layer
=================================================

This module takes the broad Top-K results from the Hybrid Retriever
and deeply analyses them using a Cross-Encoder to find the absolute best chunks.

Key responsibilities:
1. Load a local Cross-Encoder model.
2. Re-score pairs of (Query, Chunk) simultaneously.
3. Return a highly precise subset of chunks to feed to the LLM.

Design decisions:
-----------------
* Model choice: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  Why: MS-MARCO is Microsoft's massive dataset for passage ranking. This specific
  MiniLM variant is extremely fast on CPU while delivering near state-of-the-art
  accuracy for its size.
* Two-Stage Pipeline:
  We only rerank the Top 10 chunks (from config.TOP_K_RETRIEVAL). Running a 
  Cross-Encoder over the entire document would be far too slow.
"""

from typing import List, Tuple



from config import RERANKER_MODEL, TOP_K_RERANK
from core.chunker import Chunk
from utils.logger import get_logger

log = get_logger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a neural Cross-Encoder.
    """

    def __init__(self) -> None:
        """Initialise the reranker and load the model into memory."""
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            log.info(f"Loading Cross-Encoder model: {RERANKER_MODEL}")
            self.model = CrossEncoder(RERANKER_MODEL, max_length=512)
            log.info("Cross-Encoder loaded successfully.")
        except (ImportError, OSError) as e:
            log.warning(f"PyTorch/C++ Redistributables missing ({e.__class__.__name__}). Reranker disabled. Using Hybrid Retriever results directly.")

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rerank a list of candidate chunks against the query.
        
        Args:
            query: The user's original question.
            chunks: The candidate chunks (usually Top 10 from Hybrid Retriever).
            
        Returns:
            The absolute best Top N chunks (defined by TOP_K_RERANK), sorted by relevance.
        """
        if not chunks:
            log.warning("Reranker received empty chunk list.")
            return []

        if self.model is None:
            # Passthrough if reranker is disabled
            log.debug("Reranker disabled, returning Top K chunks from Hybrid Retriever.")
            return chunks[:TOP_K_RERANK]

        # The Cross-Encoder expects input as a list of pairs: [[query, text1], [query, text2], ...]
        model_inputs = [[query, chunk.text] for chunk in chunks]
        
        log.debug(f"Reranking {len(chunks)} chunks...")
        # Get relevance scores
        scores = self.model.predict(model_inputs)

        # Zip the original chunks with their new scores
        scored_chunks: List[Tuple[Chunk, float]] = list(zip(chunks, scores))
        
        # Sort descending (highest score first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Extract just the chunks, dropping the scores
        best_chunks = [chunk for chunk, score in scored_chunks[:TOP_K_RERANK]]
        
        log.info(f"Reranking complete. Selected top {len(best_chunks)} chunks.")
        
        # Log the scores of the chosen chunks for debugging
        for i, (chunk, score) in enumerate(scored_chunks[:TOP_K_RERANK]):
            log.debug(f"  Rank {i+1} score: {score:.3f} | Chunk ID: {chunk.chunk_id}")

        return best_chunks

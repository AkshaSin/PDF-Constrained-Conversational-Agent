"""
core/hybrid_retriever.py — Reciprocal Rank Fusion & Anti-Hallucination
======================================================================

This module fuses the results from the semantic (FAISS) and lexical (BM25)
retrievers into a single, unified ranking.

Key responsibilities:
1. Anti-Hallucination Layer 1: The Confidence Threshold.
   If the top FAISS result is too mathematically distant from the query, we abort
   and refuse to answer. This prevents the LLM from hallucinating an answer based
   on irrelevant text.
2. Reciprocal Rank Fusion (RRF):
   Combine FAISS and BM25 scores without needing to normalize their raw values
   (since BM25 scores and Cosine Similarities are on completely different scales).

Design decisions:
-----------------
* Why RRF? (Cormack et al., 2009)
  Trying to average a FAISS score (e.g. 0.85) with a BM25 score (e.g. 14.2) is
  statistically invalid. RRF ignores the *raw scores* and only looks at the
  *rank* (position in the list). 
  Formula: RRF_Score = 1 / (rank + k)
  This heavily rewards chunks that appear in the top 5 of *both* lists.
"""

from typing import List, Tuple, Dict

from config import (
    TOP_K_RETRIEVAL,
    RRF_CONSTANT,
    SIMILARITY_THRESHOLD_GEMINI,
    SIMILARITY_THRESHOLD_FALLBACK
)
from core.chunker import Chunk
from core.embedder import FAISSEmbedder
from core.bm25_retriever import BM25Retriever
from utils.logger import get_logger

log = get_logger(__name__)


class HybridRetriever:
    """
    Combines semantic and lexical retrieval using Reciprocal Rank Fusion.
    Enforces minimum similarity thresholds to prevent hallucination.
    """

    def __init__(self, embedder: FAISSEmbedder, bm25: BM25Retriever) -> None:
        self.embedder = embedder
        self.bm25 = bm25

    def retrieve(self, query: str) -> List[Chunk]:
        """
        Execute the hybrid retrieval pipeline.
        
        Args:
            query: The user's question.
            
        Returns:
            A list of the best chunks, or an empty list if thresholds aren't met.
        """
        log.info(f"Executing hybrid retrieval for query: '{query}'")

        # 1. Get raw results from both systems
        faiss_results = self.embedder.search(query, top_k=TOP_K_RETRIEVAL)
        bm25_results = self.bm25.search(query, top_k=TOP_K_RETRIEVAL)

        if not faiss_results:
            log.warning("No FAISS results returned.")
            return []

        # 2. Anti-Hallucination Layer 1: Confidence Threshold
        # Check the score of the absolute best FAISS result (index 0)
        top_faiss_chunk, top_faiss_score = faiss_results[0]
        
        # Determine which threshold to use based on which model is active
        threshold = (
            SIMILARITY_THRESHOLD_FALLBACK if self.embedder.using_fallback 
            else SIMILARITY_THRESHOLD_GEMINI
        )

        if top_faiss_score < threshold:
            log.warning(
                f"[ANTI-HALLUCINATION] Top FAISS score ({top_faiss_score:.3f}) "
                f"is below threshold ({threshold}). Aborting retrieval to prevent hallucination."
            )
            # Returning an empty list tells the downstream agent to respond with
            # "I don't have enough information in the document to answer that."
            return []
            
        log.debug(f"Confidence check passed. Top FAISS score: {top_faiss_score:.3f}")

        # 3. Reciprocal Rank Fusion (RRF)
        # We use chunk_id as the unique identifier to track chunks across both lists
        rrf_scores: Dict[int, float] = {}
        chunk_map: Dict[int, Chunk] = {}

        # Add FAISS ranks (1-indexed for the formula)
        for rank, (chunk, _score) in enumerate(faiss_results, start=1):
            rrf_scores[chunk.chunk_id] = 1.0 / (RRF_CONSTANT + rank)
            chunk_map[chunk.chunk_id] = chunk

        # Add BM25 ranks
        for rank, (chunk, _score) in enumerate(bm25_results, start=1):
            if chunk.chunk_id not in rrf_scores:
                rrf_scores[chunk.chunk_id] = 0.0
                chunk_map[chunk.chunk_id] = chunk
                
            rrf_scores[chunk.chunk_id] += 1.0 / (RRF_CONSTANT + rank)

        # 4. Sort by fused score
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return the absolute top chunks (up to TOP_K_RETRIEVAL)
        top_chunks = []
        for chunk_id, fused_score in fused_results[:TOP_K_RETRIEVAL]:
            top_chunks.append(chunk_map[chunk_id])
            
        log.info(f"Hybrid retrieval complete. Returning {len(top_chunks)} chunks.")
        return top_chunks

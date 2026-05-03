"""
core/bm25_retriever.py — Lexical (Keyword) Retrieval
====================================================

This module implements BM25 (Best Matching 25), the industry-standard algorithm
for sparse/keyword-based retrieval.

Key responsibilities:
1. Build an inverted index of all chunks using BM25.
2. Score chunks against a user query based on exact and partial keyword matches.

Design decisions:
-----------------
* Why BM25 alongside FAISS?
  Dense vectors (FAISS) are excellent at semantic matching ("dog" ≈ "puppy").
  However, they notoriously fail at exact matches (e.g. searching for a specific
  serial number "TX-90210" or an obscure acronym). BM25 handles the exact matches,
  and later we combine both results (Hybrid Search).
* Tokenization: Simple lowercase whitespace splitting.
  Since BM25 relies on exact term overlap, we lowercase everything to make it
  case-insensitive, and strip basic punctuation.
"""

import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from core.chunker import Chunk
from utils.logger import get_logger

log = get_logger(__name__)


class BM25Retriever:
    """
    Handles keyword-based document retrieval using the BM25 algorithm.
    """

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Chunk] = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Convert text to a list of tokens for BM25 matching.
        Lowercases text and removes non-alphanumeric characters (keeping spaces).
        """
        # Remove punctuation, make lowercase
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        # Split by whitespace
        return clean_text.split()

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Build the BM25 index from a list of chunks.
        """
        if not chunks:
            raise ValueError("No chunks provided to build BM25 index.")
            
        self.chunks = chunks
        
        # Tokenize all chunks
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        
        log.info(f"Building BM25 index with {len(chunks)} chunks...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        log.info("BM25 index built successfully.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search the BM25 index for the top_k best matching chunks.
        
        Args:
            query: The user's question.
            top_k: Number of results to return.
            
        Returns:
            List of tuples: (Chunk, BM25_score)
        """
        if self.bm25 is None or not self.chunks:
            log.warning("Search called on empty/unbuilt BM25 index.")
            return []

        tokenized_query = self._tokenize(query)
        
        # Get raw BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair up (Chunk, score), filter out 0 scores, and sort descending
        results = []
        for i, score in enumerate(scores):
            if score > 0:  # Only include chunks that actually match something
                results.append((self.chunks[i], float(score)))
                
        # Sort by score descending and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

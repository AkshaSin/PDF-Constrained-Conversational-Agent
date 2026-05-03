"""
core/embedder.py — Vector Embeddings and FAISS Indexing
=========================================================

This module transforms text chunks into dense mathematical vectors
(embeddings) and loads them into a FAISS index for high-speed similarity search.

Key responsibilities:
1. Generate embeddings using Google Gemini's API.
2. Provide a local fallback (`sentence-transformers`) if the API fails.
3. Build and query a FAISS (Facebook AI Similarity Search) index.

Design decisions:
-----------------
* Primary Model: Gemini `text-embedding-004`
  Why: State-of-the-art retrieval performance, 768 dimensions, massive free tier.
* Fallback Model: Local `all-MiniLM-L6-v2` via sentence-transformers.
  Why: If the API goes down or rate-limits, the app must not crash. MiniLM
  runs fast on CPU and gives decent results (384 dimensions).
* Index Engine: FAISS `IndexFlatIP`
  Why: "Flat" means exact nearest-neighbor search (no approximation errors).
  "IP" means Inner Product. Because our embeddings are normalised, Inner Product
  is mathematically equivalent to Cosine Similarity, which is standard for text.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import pickle
import faiss
import numpy as np
from google import genai
from google.genai import types



from config import EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL, GEMINI_API_KEY
from core.chunker import Chunk
from utils.logger import get_logger

log = get_logger(__name__)


class FAISSEmbedder:
    """
    Handles vector embedding generation and FAISS indexing.
    """

    def __init__(self) -> None:
        """Initialise embedder and prepare models."""
        # Initialize Gemini with the new genai SDK
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        self.primary_model = EMBEDDING_MODEL
        
        # Lazy-loaded fallback model (to save memory if not needed).
        # Type is Optional[Any] because SentenceTransformer is only imported
        # inside _get_fallback_model() to avoid crashing at startup when
        # PyTorch is unavailable. We cannot reference it as a class-level type hint.
        self._fallback_model: Optional[Any] = None
        self.using_fallback = False
        
        # FAISS index and metadata storage
        self.index: Optional[faiss.Index] = None
        self.chunks_map: Dict[int, Chunk] = {}
        
        # We don't know the dimension until we embed the first chunk
        self.dimension = 0

    def _get_fallback_model(self):
        """Lazy load the local sentence-transformer model."""
        if self._fallback_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                log.info(f"Loading local fallback model: {FALLBACK_EMBEDDING_MODEL}")
                self._fallback_model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
            except (ImportError, OSError) as e:
                log.error(f"Failed to load local fallback model ({e.__class__.__name__}). Ensure PyTorch and C++ redistributables are installed.")
                raise RuntimeError("Fallback model unavailable") from e
        return self._fallback_model

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts. Tries Gemini first, falls back to local model.
        Returns a normalised numpy array of shape (N, dimension).
        """
        try:
            if self.using_fallback:
                raise RuntimeError("Already switched to fallback mode.")
                
            log.debug(f"Calling Gemini API to embed {len(texts)} chunks...")
            # Gemini expects 'models/...' prefix for embeddings
            response = self.client.models.embed_content(
                model=self.primary_model,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            embeddings = np.array([e.values for e in response.embeddings], dtype=np.float32)
            
        except (RuntimeError, Exception) as e:
            if not self.using_fallback:
                log.warning(f"Gemini embedding failed: {str(e)}. Switching to local fallback.")
                self.using_fallback = True
            
            model = self._get_fallback_model()
            log.debug(f"Embedding {len(texts)} chunks with local fallback...")
            # sentence-transformers outputs normalized vectors if we ask,
            # but we'll manually normalize just to be absolutely certain
            embeddings_raw = model.encode(texts, convert_to_numpy=True)
            embeddings = np.array(embeddings_raw, dtype=np.float32)

        # FAISS Inner Product requires L2-normalised vectors to act as Cosine Similarity
        faiss.normalize_L2(embeddings)
        return embeddings

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Generate embeddings for all chunks and build the FAISS index.
        """
        if not chunks:
            raise ValueError("No chunks provided to build index.")

        texts = [chunk.text for chunk in chunks]
        
        # Embed all texts (batching logic can be added here if needed, 
        # but Gemini handles decent sized arrays natively)
        embeddings = self._embed_batch(texts)
        self.dimension = embeddings.shape[1]
        
        log.info(f"Creating FAISS IndexFlatIP with dimension {self.dimension}")
        # IndexFlatIP computes exact inner product
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        # Store metadata mapping index ID to Chunk object
        self.chunks_map = {i: chunk for i, chunk in enumerate(chunks)}
        log.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Embed the query and find the nearest chunks in the FAISS index.
        
        Args:
            query: The user's question.
            top_k: Number of results to return.
            
        Returns:
            List of tuples: (Chunk, similarity_score)
        """
        if self.index is None or self.index.ntotal == 0:
            log.warning("Search called on empty/unbuilt FAISS index.")
            return []

        # Embed the query
        try:
            if self.using_fallback:
                model = self._get_fallback_model()
                q_emb = np.array(model.encode([query]), dtype=np.float32)
            else:
                response = self.client.models.embed_content(
                    model=self.primary_model,
                    contents=[query],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                q_emb = np.array([e.values for e in response.embeddings], dtype=np.float32)
        except Exception as e:
            # If query fails, we must fallback, but query embedding must use the 
            # SAME model as the document embeddings, or spaces won't match!
            if not self.using_fallback:
                log.error("Gemini API failed during query, but index was built with Gemini! Cannot fallback mid-session without rebuilding index.")
                raise RuntimeError("Vector space mismatch: API failed during query.") from e
            raise

        faiss.normalize_L2(q_emb)

        # Perform the search
        # D = distances (cosine similarity scores), I = indices of nearest chunks
        D, I = self.index.search(q_emb, k=top_k)

        results = []
        for rank in range(len(I[0])):
            idx = I[0][rank]
            score = D[0][rank]
            if idx != -1: # FAISS returns -1 if there are fewer vectors than k
                results.append((self.chunks_map[idx], float(score)))

        return results

    def save_index(self, index_path: str, map_path: str) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        if self.index is None:
            log.warning("No index to save.")
            return
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        with open(map_path, 'wb') as f:
            pickle.dump(self.chunks_map, f)
        log.info(f"FAISS index and metadata saved to {index_path} and {map_path}.")

    def load_index(self, index_path: str, map_path: str) -> bool:
        """Load the FAISS index and chunk metadata from disk if available."""
        if os.path.exists(index_path) and os.path.exists(map_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(map_path, 'rb') as f:
                    self.chunks_map = pickle.load(f)
                self.dimension = self.index.d
                log.info(f"Loaded FAISS index with {self.index.ntotal} vectors from disk.")
                return True
            except Exception as e:
                log.error(f"Failed to load FAISS index from disk: {e}")
                self.index = None
                self.chunks_map = {}
        return False

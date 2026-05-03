"""
core/chunker.py — Text Segmentation
===================================

This module takes the single, clean string of text from `parser.py` and
segments it into overlapping chunks. 

Key responsibilities:
1. Prevent context loss by chunking the document into pieces small enough
   to fit inside the embedding model's context window.
2. Maintain continuity using a sliding window (overlap) so that sentences
   split across chunks don't lose their meaning.

Design decisions:
-----------------
* Chunking Strategy: Whitespace-based word splitting.
  Alternative considered: Character-based splitting (too naive, cuts words in half).
  Alternative considered: Semantic splitting via `spaCy` sentence boundaries.
  Why this wins for now: It maps perfectly to `config.CHUNK_SIZE` (which is in words).
  A word-based sliding window is fast, predictable, and doesn't require loading
  a heavy NLP model (like spaCy) into memory just for text splitting.
  
  (In a production v2, you would upgrade this to a semantic sentence-boundary
  chunker to ensure you never split mid-sentence).
"""

from typing import List, Tuple
from dataclasses import dataclass

from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    """
    Represents a single segment of text extracted from the document.
    Using a dataclass makes it easy to add metadata later (e.g. page numbers).
    """
    chunk_id: int
    page_number: int
    text: str
    word_count: int


class TextChunker:
    """
    Segments contiguous text into overlapping chunks.
    """

    def __init__(self, pages: List[Tuple[int, str]]) -> None:
        """
        Args:
            pages: A list of (page_number, text_content) tuples from PDFParser.
        """
        self.pages = pages
        self.words_with_pages = []
        for page_num, text in pages:
            for word in text.split():
                self.words_with_pages.append((page_num, word))
                
        log.debug(f"Initialised chunker with {len(self.words_with_pages)} words.")

    def chunk(self) -> List[Chunk]:
        """
        Execute the chunking pipeline.

        Returns:
            A list of `Chunk` objects.
        """
        if not self.words_with_pages:
            log.warning("Attempted to chunk empty text.")
            return []

        chunks: List[Chunk] = []
        chunk_id = 0
        step = CHUNK_SIZE - CHUNK_OVERLAP

        # Ensure step is strictly positive to prevent infinite loops
        if step <= 0:
            raise ValueError(
                f"CHUNK_SIZE ({CHUNK_SIZE}) must be greater than "
                f"CHUNK_OVERLAP ({CHUNK_OVERLAP})"
            )

        # Sliding window over the word list
        for i in range(0, len(self.words_with_pages), step):
            # Take a slice of CHUNK_SIZE words
            chunk_slice = self.words_with_pages[i : i + CHUNK_SIZE]
            
            page_number = chunk_slice[0][0]
            chunk_words = [word for _, word in chunk_slice]
            
            # Reconstruct the string for this chunk
            chunk_text = " ".join(chunk_words)
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                page_number=page_number,
                text=chunk_text,
                word_count=len(chunk_words)
            ))
            
            chunk_id += 1

        log.info(
            f"Segmented {len(self.words_with_pages)} words into {len(chunks)} chunks "
            f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
        )
        
        return chunks

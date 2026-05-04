from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class DocumentMetadata:
    """
    Stores document-level factual metadata that is computed once during ingestion.
    This allows the FactualAnswerer to respond to statistical queries in O(1) time
    without running a full-document text scan or relying on the LLM's hallucination-prone memory.
    """
    page_count: int
    word_count: int
    char_count: int
    word_frequencies: Dict[str, int]
    # List of character offsets indicating where each page starts within full_text.
    # Contract: page_boundaries[i] is the character index in full_text where page (i+1) begins.
    # The final boundary is implicitly len(full_text).
    page_boundaries: List[int]
    # The complete, concatenated text of the document used for precise regex scanning and tool extraction.
    full_text: str

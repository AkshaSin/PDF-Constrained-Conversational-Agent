"""
core/parser.py — PDF Ingestion and Cleaning
=============================================

This module is responsible for turning a raw PDF file into a clean, contiguous
string of text, ready for chunking.

Key responsibilities:
1. Extract raw text from PDF bytes.
2. Remove repeating headers and footers (which degrade search quality).
3. Normalise whitespace and fix common PDF extraction artifacts (e.g.,
   broken hyphenation across lines).

Design decisions:
-----------------
* Library choice: PyMuPDF (fitz)
  Alternative: pdfplumber (more precise bounding boxes, but 5-10x slower).
  Alternative: PyPDF2 (very fast, but poor text layout retention).
  Why PyMuPDF wins: It's the industry standard for fast, high-quality text
  extraction at scale.

* Header/Footer Removal Strategy:
  We use a heuristic approach based on frequency analysis. If a short line of
  text (e.g., "Company Confidential" or "Page 4") appears on >60% of pages,
  we classify it as a header/footer and strip it.
  Alternative: Training an ML model to classify page regions. Too complex,
  slow, and brittle for this scale. Frequency analysis is robust and fast.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

import fitz  # PyMuPDF

from config import HEADER_FOOTER_FREQ_THRESHOLD, HEADER_FOOTER_MIN_PAGES
from utils.logger import get_logger

log = get_logger(__name__)


class PDFParser:
    """
    Handles extraction and cleaning of text from PDF files.
    """

    def __init__(self, file_bytes: bytes) -> None:
        """
        Initialise parser with raw PDF bytes.
        
        Args:
            file_bytes: The raw binary content of the PDF file.
        """
        self.file_bytes = file_bytes
        self.doc = fitz.open(stream=file_bytes, filetype="pdf")
        self.num_pages = len(self.doc)
        log.info(f"Loaded PDF with {self.num_pages} pages.")

    def _extract_raw_pages(self) -> List[str]:
        """Extract raw text from each page."""
        pages = []
        for i in range(self.num_pages):
            page = self.doc[i]
            # get_text("text") returns text ordered roughly by human reading order
            pages.append(page.get_text("text"))
        return pages

    def _find_headers_and_footers(self, pages: List[str]) -> set[str]:
        """
        Identify recurring short lines across pages that are likely headers/footers.
        """
        if self.num_pages < HEADER_FOOTER_MIN_PAGES:
            return set()

        line_counter = Counter()
        
        for page_text in pages:
            lines = page_text.split('\n')
            # Deduplicate within a single page so a phrase repeated on the same
            # page doesn't skew the cross-page frequency.
            unique_lines = set(line.strip() for line in lines if line.strip())
            
            for line in unique_lines:
                # Heuristic: headers/footers are usually short (< 10 words)
                if len(line.split()) < 10:
                    line_counter[line] += 1

        threshold = int(self.num_pages * HEADER_FOOTER_FREQ_THRESHOLD)
        
        noise_lines = {
            line for line, count in line_counter.items() 
            if count >= threshold
        }
        
        if noise_lines:
            log.debug(f"Detected {len(noise_lines)} repeating headers/footers.")
            
        return noise_lines

    def _clean_text(self, pages: List[str], noise_lines: set[str]) -> str:
        """
        Remove noise lines and normalise text.
        """
        clean_pages = []
        
        for page_text in pages:
            lines = page_text.split('\n')
            clean_lines = []
            
            for line in lines:
                stripped = line.strip()
                # 1. Drop empty lines
                if not stripped:
                    continue
                # 2. Drop headers/footers
                if stripped in noise_lines:
                    continue
                # 3. Drop standalone page numbers (digits only)
                if stripped.isdigit():
                    continue
                    
                clean_lines.append(stripped)
                
            # Join lines with a space, but we'll need to fix hyphens later
            clean_pages.append(" ".join(clean_lines))

        # Join all pages
        full_text = " ".join(clean_pages)
        
        # --- Post-processing normalisation ---
        
        # Fix broken hyphenation: "pro- ject" -> "project"
        # Often happens when a word wraps across a line in a PDF.
        full_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', full_text)
        
        # Collapse multiple spaces into one
        full_text = re.sub(r'\s+', ' ', full_text)
        
        return full_text.strip()

    def parse(self) -> str:
        """
        Execute the full parsing pipeline.
        
        Returns:
            A single, cleaned string containing the entire document's text.
        """
        try:
            pages = self._extract_raw_pages()
            noise_lines = self._find_headers_and_footers(pages)
            clean_text = self._clean_text(pages, noise_lines)
            
            log.info(f"Successfully parsed PDF. Extracted {len(clean_text)} characters.")
            return clean_text
            
        except Exception as e:
            log.error(f"Failed to parse PDF: {str(e)}")
            raise
        finally:
            # Always close the PyMuPDF document to free C-level memory
            self.doc.close()

"""
core/tools.py — Agentic Tool Registry and Execution
===================================================

Defines the Python tools available to the LLM for performing deterministic
document calculations (e.g., word counting, exact snippet extraction).
"""

import re
import bisect
from dataclasses import dataclass
from typing import Callable, Any, List, Dict
from core.metadata_store import DocumentMetadata
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable


@dataclass
class ToolResult:
    tool_name: str
    output: Any = None
    error: str = None


# --- Tool Handlers ---

def _count_word(word: str, full_text: str) -> int:
    """Exact case-insensitive count of word occurrences."""
    pattern = r'\b' + re.escape(word) + r'\b'
    matches = re.findall(pattern, full_text, re.IGNORECASE)
    return len(matches)


def _get_page_count(full_text: str, page_boundaries: List[int]) -> int:
    """Returns total number of pages."""
    return len(page_boundaries)


def _get_page_content(page_num: int, full_text: str, page_boundaries: List[int]) -> str:
    """Returns the text slice for that page number (1-indexed)."""
    if not page_boundaries:
        return ""
    if page_num < 1 or page_num > len(page_boundaries):
        raise ValueError(f"Page number {page_num} is out of bounds (1-{len(page_boundaries)}).")
        
    start_idx = page_boundaries[page_num - 1]
    end_idx = page_boundaries[page_num] if page_num < len(page_boundaries) else len(full_text)
    
    return full_text[start_idx:end_idx].strip()


def _find_all_occurrences(phrase: str, full_text: str, page_boundaries: List[int]) -> List[Dict[str, Any]]:
    """Returns list of {page: int, snippet: str} for every match."""
    if not phrase.strip():
        return []
        
    results = []
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    
    for match in pattern.finditer(full_text):
        start = match.start()
        end = match.end()
        
        # Clamp slice with max/min to avoid boundary crashes
        snippet_start = max(0, start - 40)
        snippet_end = min(len(full_text), end + 40)
        snippet = full_text[snippet_start:snippet_end]
        
        # Clean up snippet boundaries if truncated
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(full_text):
            snippet = snippet + "..."
            
        # Determine page number using bisect
        # bisect_right gives the insertion point. Since boundaries are start offsets,
        # the page index is bisect_right(start) - 1.
        # Adding 1 makes it 1-indexed.
        page_idx = bisect.bisect_right(page_boundaries, start)
        page_num = page_idx if page_idx > 0 else 1
        
        results.append({
            "page": page_num,
            "snippet": snippet.strip().replace('\n', ' ')
        })
        
    return results


class ToolExecutor:
    """
    Holds the registry of available tools and safely executes them.
    """
    
    def __init__(self, metadata: DocumentMetadata):
        self.metadata = metadata
        self.tools: Dict[str, Tool] = {}
        self._register_tools()
        
    def _register_tools(self):
        self.tools["count_word"] = Tool(
            name="count_word",
            description="Count exact occurrences of a word in the document.",
            parameters={"word": "The exact word to count (case-insensitive)"},
            handler=_count_word
        )
        self.tools["get_page_count"] = Tool(
            name="get_page_count",
            description="Get the total number of pages in the document.",
            parameters={},
            handler=_get_page_count
        )
        self.tools["get_page_content"] = Tool(
            name="get_page_content",
            description="Retrieve the exact full text of a specific page.",
            parameters={"page_num": "The 1-indexed page number to retrieve"},
            handler=_get_page_content
        )
        self.tools["find_all_occurrences"] = Tool(
            name="find_all_occurrences",
            description="Find all occurrences of a specific phrase and return their page numbers and context snippets.",
            parameters={"phrase": "The exact phrase to search for (case-insensitive)"},
            handler=_find_all_occurrences
        )
        
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name, automatically injecting metadata parameters."""
        log.info(f"Executing tool: {tool_name} with args: {kwargs}")
        
        if tool_name not in self.tools:
            return ToolResult(tool_name=tool_name, error=f"Unknown tool: {tool_name}")
            
        tool = self.tools[tool_name]
        
        # Inject metadata
        if tool_name in ["count_word"]:
            kwargs["full_text"] = self.metadata.full_text
        elif tool_name in ["get_page_count", "get_page_content", "find_all_occurrences"]:
            kwargs["full_text"] = self.metadata.full_text
            kwargs["page_boundaries"] = self.metadata.page_boundaries
            
        try:
            result = tool.handler(**kwargs)
            return ToolResult(tool_name=tool_name, output=result)
        except TypeError as e:
            err_msg = f"Invalid arguments for {tool_name}: {str(e)}"
            log.error(err_msg)
            return ToolResult(tool_name=tool_name, error=err_msg)
        except Exception as e:
            err_msg = f"Error executing {tool_name}: {str(e)}"
            log.error(err_msg)
            return ToolResult(tool_name=tool_name, error=err_msg)

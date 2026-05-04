"""
tests/test_thinking_transforms.py — Unit Tests for Thinking Tag Transforms
===========================================================================

Tests for the two pure helper functions added to app.py:
  - transform_thinking_tags(text) → Converts <thinking> tags to <details> accordions
  - strip_thinking(text)         → Strips <thinking> blocks before Redis memory storage
"""

import sys
import os
import re
import unittest

# ---------------------------------------------------------------------------
# Pull the two helpers out of app.py without importing the full Gradio app
# (which would trigger backend init, Redis connections, etc.)
# ---------------------------------------------------------------------------

def transform_thinking_tags(text: str) -> str:
    """Inline copy of the helper — kept in sync with app.py."""
    # Guard: closing tag partially streamed — check ALL partial prefixes
    _CLOSING = "</thinking>"
    if any(_CLOSING[:i] in text for i in range(2, len(_CLOSING))) and _CLOSING not in text:
        return text
    # Guard: opening tag partially streamed (e.g. "<thinki" or "<thinking" without ">")
    if "<thinking" in text and "<thinking>" not in text:
        return text
    text = text.replace(
        "<thinking>",
        "<details open>\n<summary>\U0001f9e0 Agent Thinking...</summary>\n\n"
    )
    text = text.replace("</thinking>", "\n</details>\n\n")
    return text


def strip_thinking(text: str) -> str:
    """Inline copy of the helper — kept in sync with app.py."""
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()


# ===========================================================================
# Tests for transform_thinking_tags
# ===========================================================================

class TestTransformThinkingTags(unittest.TestCase):

    def test_complete_tags_converted(self):
        """Complete <thinking>...</thinking> is replaced with <details open>."""
        result = transform_thinking_tags("<thinking>foo</thinking>bar")
        self.assertIn("<details open>", result)
        self.assertIn("</details>", result)
        self.assertIn("bar", result)
        self.assertNotIn("<thinking>", result)
        self.assertNotIn("</thinking>", result)

    def test_partial_opening_tag_unchanged(self):
        """Partial opening tag (no closing >) must be returned unchanged."""
        partial = "<thinki"
        self.assertEqual(transform_thinking_tags(partial), partial)

    def test_partial_opening_tag_with_content_unchanged(self):
        """'<thinking' without '>' is partial — must be returned unchanged."""
        partial = "<thinking"
        self.assertEqual(transform_thinking_tags(partial), partial)

    def test_partial_closing_tag_unchanged(self):
        """Partial closing tag must not produce broken HTML."""
        partial = "<thinking>foo</thinki"
        self.assertEqual(transform_thinking_tags(partial), partial)

    def test_partial_closing_tag_no_gt_unchanged(self):
        """'</thinking' without '>' is partial — must be returned unchanged."""
        partial = "<thinking>foo</thinking"
        self.assertEqual(transform_thinking_tags(partial), partial)

    def test_empty_string(self):
        """Empty input returns empty string."""
        self.assertEqual(transform_thinking_tags(""), "")

    def test_no_tags_passthrough(self):
        """Text with no thinking tags is returned unchanged."""
        text = "just an answer"
        self.assertEqual(transform_thinking_tags(text), text)

    def test_phrase_with_thinking_word_unchanged(self):
        """'thinking' as a natural word (no tag) must not be altered."""
        text = "I was thinking about it"
        self.assertEqual(transform_thinking_tags(text), text)

    def test_tool_call_response_unchanged(self):
        """A raw tool call response must never be altered."""
        text = '<tool_call>{"name": "get_page_count", "args": {}}</tool_call>'
        self.assertEqual(transform_thinking_tags(text), text)

    def test_summary_label_present(self):
        """The accordion summary label must contain the 🧠 emoji."""
        result = transform_thinking_tags("<thinking>reasoning</thinking>answer")
        self.assertIn("\U0001f9e0 Agent Thinking...", result)

    def test_details_open_attribute(self):
        """Accordion must use <details open> so it auto-expands during streaming."""
        result = transform_thinking_tags("<thinking>r</thinking>a")
        self.assertIn("<details open>", result)

    def test_multiple_thinking_blocks(self):
        """Multiple thinking blocks are all converted."""
        text = "<thinking>r1</thinking>a1<thinking>r2</thinking>a2"
        result = transform_thinking_tags(text)
        self.assertEqual(result.count("<details open>"), 2)
        self.assertEqual(result.count("</details>"), 2)


# ===========================================================================
# Tests for strip_thinking
# ===========================================================================

class TestStripThinking(unittest.TestCase):

    def test_strips_complete_block(self):
        """A complete <thinking>...</thinking> block is removed."""
        text = "<thinking>my reasoning</thinking>Final answer."
        self.assertEqual(strip_thinking(text), "Final answer.")

    def test_strips_block_with_newlines(self):
        """Multi-line thinking blocks are fully stripped."""
        text = "<thinking>\nline one\nline two\n</thinking>\nFinal answer."
        self.assertEqual(strip_thinking(text), "Final answer.")

    def test_no_thinking_block_unchanged(self):
        """Text with no thinking block is returned as-is."""
        text = "Just a plain answer."
        self.assertEqual(strip_thinking(text), text)

    def test_strips_multiple_thinking_blocks(self):
        """Multiple thinking blocks (from multi-turn if ever concatenated) are all removed."""
        text = "<thinking>r1</thinking>A1. <thinking>r2</thinking>A2."
        result = strip_thinking(text)
        self.assertNotIn("<thinking>", result)
        self.assertNotIn("</thinking>", result)
        self.assertIn("A1.", result)
        self.assertIn("A2.", result)

    def test_leading_trailing_whitespace_stripped(self):
        """Result is trimmed of leading/trailing whitespace."""
        text = "<thinking>r</thinking>   answer   "
        self.assertEqual(strip_thinking(text), "answer")

    def test_empty_thinking_block(self):
        """An empty thinking block is stripped cleanly."""
        text = "<thinking></thinking>answer"
        self.assertEqual(strip_thinking(text), "answer")

    def test_empty_string(self):
        """Empty input returns empty string."""
        self.assertEqual(strip_thinking(""), "")


if __name__ == "__main__":
    unittest.main()

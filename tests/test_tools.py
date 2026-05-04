import unittest
from core.tools import _count_word, _get_page_count, _get_page_content, _find_all_occurrences, ToolExecutor
from core.metadata_store import DocumentMetadata

class TestTools(unittest.TestCase):
    def setUp(self):
        self.full_text = "Hello world. This is a test document. The word 'test' appears twice. No wait, testing doesn't count. Test!"
        self.page_boundaries = [0, 50, 100] # Let's say 3 pages

    def test_count_word(self):
        # Case insensitive exact match.
        # "test" -> "test document", "word 'test'", "Test!" -> 3 matches
        # "testing" does not count.
        count = _count_word("test", self.full_text)
        self.assertEqual(count, 3)
        
        # Word not in text
        self.assertEqual(_count_word("banana", self.full_text), 0)

    def test_get_page_count(self):
        self.assertEqual(_get_page_count(3), 3)

    def test_get_page_content(self):
        # bounds: 0, 50, 100
        # len is 108
        page_1 = _get_page_content(1, self.full_text, self.page_boundaries)
        page_2 = _get_page_content(2, self.full_text, self.page_boundaries)
        page_3 = _get_page_content(3, self.full_text, self.page_boundaries)
        
        self.assertEqual(page_1, self.full_text[0:50].strip())
        self.assertEqual(page_2, self.full_text[50:100].strip())
        self.assertEqual(page_3, self.full_text[100:108].strip())
        
        # Out of bounds
        with self.assertRaises(ValueError):
            _get_page_content(0, self.full_text, self.page_boundaries)
        with self.assertRaises(ValueError):
            _get_page_content(4, self.full_text, self.page_boundaries)

    def test_find_all_occurrences(self):
        results = _find_all_occurrences("test", self.full_text, self.page_boundaries)
        self.assertEqual(len(results), 4) # "test document", "test' appears", "testing doesn't", "Test!"
        
        # Check snippet truncation and page numbers
        # "test document" is around index 23 -> page 1
        self.assertEqual(results[0]["page"], 1)
        self.assertTrue("test" in results[0]["snippet"].lower())

    def test_tool_executor(self):
        metadata = DocumentMetadata(3, 10, 108, {}, self.page_boundaries, self.full_text)
        executor = ToolExecutor(metadata)
        
        result = executor.execute("count_word", word="test")
        self.assertEqual(result.output, 3)
        self.assertIsNone(result.error)
        
        # Unknown tool
        result2 = executor.execute("fake_tool")
        self.assertIsNotNone(result2.error)

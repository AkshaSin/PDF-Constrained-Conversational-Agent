import unittest
from unittest.mock import patch, MagicMock
from agent.generator import LLMGenerator
from core.metadata_store import DocumentMetadata
from core.tools import ToolExecutor

class MockChunk:
    def __init__(self, text):
        self.text = text

class MockStream:
    def __init__(self, chunks):
        self.chunks = chunks
        
    def __iter__(self):
        for c in self.chunks:
            yield MockChunk(c)

class TestGenerateWithTools(unittest.TestCase):
    def setUp(self):
        self.generator = LLMGenerator()
        self.metadata = DocumentMetadata(1, 10, 100, {}, [0], "fake text")
        self.executor = ToolExecutor(self.metadata)

    @patch('google.genai.Client')
    def test_no_tool_call(self, mock_client):
        # Mock LLM returning a direct answer
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content_stream.return_value = MockStream(["Hello ", "world!"])
        
        stream = self.generator.generate_with_tools("query", [], [], self.executor)
        
        results = list(stream)
        self.assertEqual("".join(results), "Hello world!")

    @patch('google.genai.Client')
    def test_with_tool_call(self, mock_client):
        # First iteration returns a tool call
        # Second iteration returns the final answer
        mock_instance = mock_client.return_value
        
        # We need side_effect to return different streams on each call
        mock_instance.models.generate_content_stream.side_effect = [
            MockStream(['<tool_call>{"name": "count_word", "args": {"word": "test"}}</tool_call>']),
            MockStream(["The word test appears ", "3 times."])
        ]
        
        # Override executor to return a fake result without actually running re.findall on fake text
        self.executor.execute = MagicMock()
        self.executor.execute.return_value = MagicMock(error=None, output=3)
        
        stream = self.generator.generate_with_tools("query", [], [], self.executor)
        
        results = list(stream)
        
        # The first item should be a tool status dict
        self.assertTrue(isinstance(results[0], dict))
        self.assertEqual(results[0]["type"], "tool_status")
        self.assertEqual(results[0]["name"], "count_word")
        
        # The remaining items should be strings
        final_answer = "".join(results[1:])
        self.assertEqual(final_answer, "The word test appears 3 times.")
        
        # Ensure LLM was called twice
        self.assertEqual(mock_instance.models.generate_content_stream.call_count, 2)

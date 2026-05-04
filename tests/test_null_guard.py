import unittest
from unittest.mock import patch, MagicMock
from app import chat_inference

class TestNullGuard(unittest.TestCase):
    @patch('app.SessionMemory')
    def test_null_metadata_aborts(self, mock_session_memory_cls):
        mock_memory = MagicMock()
        mock_memory.get_metadata.return_value = None
        mock_session_memory_cls.return_value = mock_memory

        generator = chat_inference("what is the page count?", [], "test_session")
        first_ui_update = next(generator) # The chat_history.append update (user, empty assistant)
        response = next(generator) # The actual null guard yield
        
        self.assertEqual(len(response), 3) # user, empty assistant, system warning
        self.assertEqual(response[-1]["role"], "assistant")
        self.assertIn("Document metadata not found", response[-1]["content"])

if __name__ == '__main__':
    unittest.main()

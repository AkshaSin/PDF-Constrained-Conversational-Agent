import unittest
from unittest.mock import patch, MagicMock
from core.query_router import QueryRouter

class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()

    def test_page_count_regex(self):
        res = self.router.classify("how many pages are in this document?")
        self.assertEqual(res["intent"], "factual")
        self.assertEqual(res["sub_intent"], "page_count")
        self.assertEqual(res["confidence"], 1.0)

    def test_word_count_regex(self):
        res = self.router.classify("how many times is the word 'climate' used?")
        self.assertEqual(res["intent"], "factual")
        self.assertEqual(res["sub_intent"], "word_count:climate")
        self.assertEqual(res["confidence"], 1.0)

    def test_aggregation_regex(self):
        res = self.router.classify("can you list all instances of renewable energy?")
        self.assertEqual(res["intent"], "aggregation")
        self.assertIsNone(res["sub_intent"])
        self.assertEqual(res["confidence"], 1.0)

    def test_top_words_regex(self):
        res = self.router.classify("what are the top 5 most used words?")
        self.assertEqual(res["intent"], "factual")
        self.assertEqual(res["sub_intent"], "top_words:5")
        self.assertEqual(res["confidence"], 1.0)

    @patch('core.query_router.genai.Client')
    def test_explanation_semantic_llm(self, mock_client):
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.text = '{"intent": "semantic", "reasoning": "Asking for explanation"}'
        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response
        mock_instance = MagicMock()
        mock_instance.models = mock_models
        mock_client.return_value = mock_instance

        res = self.router.classify("Explain the greenhouse effect based on the text.")
        self.assertEqual(res["intent"], "semantic")
        self.assertEqual(res["confidence"], 0.5)

    @patch('core.query_router.genai.Client')
    def test_ambiguous_query_default(self, mock_client):
        # Mock LLM returning invalid format or unsupported intent
        mock_response = MagicMock()
        mock_response.text = '{"intent": "unknown"}'
        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response
        mock_instance = MagicMock()
        mock_instance.models = mock_models
        mock_client.return_value = mock_instance

        res = self.router.classify("what?")
        self.assertEqual(res["intent"], "semantic")
        self.assertEqual(res["confidence"], 0.5)

    @patch('core.query_router.genai.Client')
    def test_hindi_query_semantic(self, mock_client):
        mock_response = MagicMock()
        mock_response.text = '{"intent": "semantic", "reasoning": "Hindi explanation"}'
        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response
        mock_instance = MagicMock()
        mock_instance.models = mock_models
        mock_client.return_value = mock_instance

        res = self.router.classify("इस दस्तावेज़ में क्या है?")
        self.assertEqual(res["intent"], "semantic")

    @patch('core.query_router.genai.Client')
    def test_french_query_semantic(self, mock_client):
        mock_response = MagicMock()
        mock_response.text = '{"intent": "semantic", "reasoning": "French explanation"}'
        mock_models = MagicMock()
        mock_models.generate_content.return_value = mock_response
        mock_instance = MagicMock()
        mock_instance.models = mock_models
        mock_client.return_value = mock_instance

        res = self.router.classify("Quel est le sujet principal ?")
        self.assertEqual(res["intent"], "semantic")

if __name__ == '__main__':
    unittest.main()

from core.metadata_store import DocumentMetadata
from utils.logger import get_logger

log = get_logger(__name__)

class FactualAnswerer:
    """
    Handles factual query intents by looking up pre-computed metadata from the DocumentMetadata.
    Bypasses standard semantic retrieval for exact statistical queries.
    """
    def __init__(self, metadata: DocumentMetadata):
        self.metadata = metadata

    def answer(self, sub_intent: str, query: str) -> str:
        """
        Routes the sub_intent to the correct factual method.
        """
        if not sub_intent:
            return "I could not determine the exact factual statistic you were asking for."

        if sub_intent == "page_count":
            return self.get_page_count()
            
        if sub_intent.startswith("word_count:"):
            word = sub_intent.split(":", 1)[1].strip()
            return self.count_word(word)
            
        if sub_intent.startswith("top_words:"):
            try:
                n = int(sub_intent.split(":", 1)[1].strip())
            except ValueError:
                n = 5
            return self.get_word_frequency_top_n(n)
            
        return "The requested statistic is not supported by the factual answerer."

    def get_page_count(self) -> str:
        return f"The document has {self.metadata.page_count} pages."

    def count_word(self, word: str) -> str:
        word_lower = word.lower()
        count = self.metadata.word_frequencies.get(word_lower, 0)
        return f"The word '{word}' is used {count} times in the document."

    def get_word_frequency_top_n(self, n: int) -> str:
        sorted_words = sorted(self.metadata.word_frequencies.items(), key=lambda item: item[1], reverse=True)
        # Filter out extremely common short stop words to make the result more meaningful
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "as", "is", "are", "was", "were", "it", "this", "that"}
        filtered = [(w, c) for w, c in sorted_words if w not in stop_words and len(w) > 2]
        
        top_n = filtered[:n]
        if not top_n:
            return "Could not determine the top words."
            
        result = [f"The top {len(top_n)} most used words (excluding common stop words) are:"]
        for i, (w, c) in enumerate(top_n, 1):
            result.append(f"{i}. '{w}' ({c} times)")
            
        return "\n".join(result)

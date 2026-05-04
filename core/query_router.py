import re
from typing import TypedDict, Optional
from google import genai
from google.genai import types
from config import GENERATION_MODEL, GEMINI_API_KEY
from utils.logger import get_logger

log = get_logger(__name__)

class RouterResult(TypedDict):
    intent: str
    sub_intent: Optional[str]
    confidence: float
    reasoning: str

ROUTER_PROMPT = """You are a query classification router. Classify the user's query into exactly ONE of these intents:
1. "semantic": Conceptual questions, explanations, summaries, or questions about the content meaning.
2. "factual": Questions about document statistics, exact page counts, or exact word frequencies.
3. "aggregation": Requests to "list all instances", "find every mention", or aggregate information across the entire document.

Respond ONLY with a JSON object in this format:
{"intent": "semantic" | "factual" | "aggregation", "reasoning": "brief explanation"}

User Query: {query}"""

class QueryRouter:
    """
    Classifies an incoming query into semantic, factual, or aggregation paths.
    Uses regex for high-confidence structural queries (O(1) latency), and falls back
    to an LLM call if the query is ambiguous.
    """
    def __init__(self):
        self._api_key = GEMINI_API_KEY

    def classify(self, query: str) -> RouterResult:
        query_lower = query.lower()
        
        # 1. Regex Rules (High Confidence)
        if re.search(r'\bhow many pages\b', query_lower):
            return {"intent": "factual", "sub_intent": "page_count", "confidence": 1.0, "reasoning": "Regex match for page count"}
        
        word_count_match = re.search(r'\bhow many times is the word ["\']?(\w+)["\']? used\b', query_lower)
        if word_count_match:
            return {"intent": "factual", "sub_intent": f"word_count:{word_count_match.group(1)}", "confidence": 1.0, "reasoning": "Regex match for word count"}
            
        freq_match = re.search(r'\bwhat are the top (\d+) most used words\b', query_lower)
        if freq_match:
             return {"intent": "factual", "sub_intent": f"top_words:{freq_match.group(1)}", "confidence": 1.0, "reasoning": "Regex match for top words"}

        if re.search(r'\b(list all|find every mention of|find all instances of)\b', query_lower):
            return {"intent": "aggregation", "sub_intent": None, "confidence": 1.0, "reasoning": "Regex match for aggregation"}
            
        # 2. LLM Fallback
        return self._llm_fallback(query)

    def _llm_fallback(self, query: str) -> RouterResult:
        log.debug("Regex classification failed. Falling back to LLM intent routing.")
        try:
            client = genai.Client(api_key=self._api_key)
            config = types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=ROUTER_PROMPT.replace("{query}", query),
                config=config
            )
            
            # Simple JSON parsing fallback, Gemini JSON mode returns valid JSON
            import json
            data = json.loads(response.text)
            intent = data.get("intent", "semantic")
            reasoning = data.get("reasoning", "LLM classified")
            
            # If the LLM returned something unexpected, default to semantic
            if intent not in ["semantic", "factual", "aggregation"]:
                intent = "semantic"
                
            return {
                "intent": intent, 
                "sub_intent": None, # Complex factual intents via LLM fallback not yet mapped to sub_intents
                "confidence": 0.5, # Lower confidence for LLM heuristic
                "reasoning": reasoning
            }
        except Exception as e:
            log.warning(f"LLM routing failed: {e}. Defaulting to semantic.")
            return {"intent": "semantic", "sub_intent": None, "confidence": 0.0, "reasoning": "Fallback due to LLM error"}

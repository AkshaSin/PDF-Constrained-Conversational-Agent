import re
from typing import TypedDict, Optional
from google import genai
from google.genai import types
from config import ROUTER_MODEL, GEMINI_API_KEY
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

    # Patterns for word-count queries. Each captures the target word in group 1.
    _WORD_COUNT_PATTERNS = [
        re.compile(r'\bhow many times (?:is|does|was|has) (?:the word )?["\']?(\w+)["\']? (?:used|appear|mentioned|occur|come up)\b'),
        re.compile(r'\bhow many times (?:does )?(?:the word )?["\']?(\w+)["\']? (?:appear|occur|comes up)\b'),
        re.compile(r'\bcount (?:the word |occurrences of |instances of )?["\']?(\w+)["\']?\b'),
        re.compile(r'\bhow often (?:is|does) (?:the word )?["\']?(\w+)["\']? (?:used|appear|mentioned|occur)\b'),
        re.compile(r'\b(?:frequency|occurrences|instances) of ["\']?(\w+)["\']?\b'),
        re.compile(r'\bhow many (?:times|occurrences|instances) (?:of )?["\']?(\w+)["\']?\b'),
    ]

    def classify(self, query: str) -> RouterResult:
        query_lower = query.lower()

        # 1. Regex Rules (High Confidence)
        if re.search(r'\b(how many pages|page count|number of pages|total pages|pages (does|in (the|this)))\b', query_lower):
            return {"intent": "factual", "sub_intent": "page_count", "confidence": 1.0, "reasoning": "Regex match for page count"}

        for pattern in self._WORD_COUNT_PATTERNS:
            m = pattern.search(query_lower)
            if m:
                word = m.group(1)
                return {"intent": "factual", "sub_intent": f"word_count:{word}", "confidence": 1.0, "reasoning": "Regex match for word count"}

        freq_match = re.search(r'\b(?:what are |show me |list )?the top (\d+) (?:most )?(?:used |common |frequent )?words\b', query_lower)
        if freq_match:
             return {"intent": "factual", "sub_intent": f"top_words:{freq_match.group(1)}", "confidence": 1.0, "reasoning": "Regex match for top words"}

        if re.search(r'\b(list all|find every mention of|find all instances of|all occurrences of)\b', query_lower):
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
                model=ROUTER_MODEL,
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

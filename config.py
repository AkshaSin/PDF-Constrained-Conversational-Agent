"""
config.py — Central Configuration Module
=========================================

Single source of truth for all constants, model names, thresholds, and
credentials used across the PDF-Constrained Conversational Agent.

Design principle: FAIL-FAST validation.
  - All required environment variables are validated at import time.
  - The app will crash immediately with a clear error message if any are missing.
  - This prevents mid-demo failures and ensures the system is always in a
    known-good state before any user interaction begins.

Usage:
    from config import GEMINI_API_KEY, CHUNK_SIZE, TOP_K_RETRIEVAL, ...

Environment variables (set in .env or HF Spaces Secrets):
    GEMINI_API_KEY  — Google Gemini API key
    REDIS_URL       — Redis connection string (e.g. redis://default:pass@host:port)
"""

import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file into os.environ (no-op if already set via system env, which
# means HF Spaces secrets work without any code change at deployment time).
# ---------------------------------------------------------------------------
load_dotenv()


def _require_env(key: str) -> str:
    """
    Retrieve a required environment variable, raising at import time if missing.

    This implements the fail-fast pattern: rather than returning None and letting
    the error surface later (possibly mid-demo), we crash immediately with an
    actionable message pointing the user to the exact missing variable.

    Args:
        key: The environment variable name to look up.

    Returns:
        The value of the environment variable as a string.

    Raises:
        EnvironmentError: If the variable is not set or is an empty string.
    """
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"[config.py] Required environment variable '{key}' is missing or empty.\n"
            f"  → Add it to your .env file (local) or HF Spaces Secrets (deployment).\n"
            f"  → See .env.example for the expected format."
        )
    return value


# ---------------------------------------------------------------------------
# CREDENTIALS (fail-fast — these must be set before the app starts)
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = _require_env("GEMINI_API_KEY")
"""Google Gemini API key. Used for embeddings, generation, and faithfulness verification."""

REDIS_URL: str = _require_env("REDIS_URL")
"""
Redis connection URL. Format: redis://default:<password>@<host>:<port>
Used for session history storage and PDF index caching.
"""


# ---------------------------------------------------------------------------
# MODEL NAMES
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "models/text-embedding-004"
"""
Primary embedding model (Gemini). Free tier, 768-dimensional output.
Chosen over OpenAI ada-002 (paid) and older Gemini models (lower quality).
"""

GENERATION_MODEL: str = "gemini-1.5-flash"
"""
Generation model. 1M context window, free tier, fast streaming.
Alternative considered: gemini-1.5-pro (higher quality but slower, lower free quota).
"""

RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
"""
Cross-encoder reranker. Runs locally (no API calls, no cost).
ms-marco is the standard passage reranking dataset; MiniLM-L-6-v2 gives the best
speed/accuracy trade-off at this scale. Alternative: ms-marco-MiniLM-L-12-v2
(more accurate, 2× slower — overkill for <10 candidates).
"""

FALLBACK_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
"""
Local sentence-transformers fallback for when Gemini embedding API is unavailable.
384-dimensional output — note the different dimension from Gemini (768).
IMPORTANT: If fallback is used, it must be used consistently for ALL vectors
in that session. Mixing models in one FAISS index produces garbage similarities.
See core/embedder.py for the model-consistency enforcement logic.
"""


# ---------------------------------------------------------------------------
# CHUNKING PARAMETERS
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 500
"""
Chunk size in whitespace-tokenised words (1 token ≈ 1 word for chunking purposes).

Rationale: 500 words ≈ 650 sub-word tokens on average, comfortably within the
512-token limit of most bi-encoders, and small enough that a chunk maps clearly
to a specific topic/section.

Alternative considered: 256 words (finer granularity, more chunks, higher recall
but more noise). 1000 words (fewer chunks, lower latency but worse precision —
a single chunk may span multiple unrelated topics, confusing the reranker).

v1 plan: Replace with semantic chunking using spaCy sentence boundaries.
See core/chunker.py for implementation details.
"""

CHUNK_OVERLAP: int = 50
"""
Overlap between consecutive chunks in words.

Preserves concept continuity at chunk boundaries. A sentence split across two
chunks will appear in full in at least one of them.
Reference: Module 7 Page 6 — fixed vs overlapping chunking windows.
"""


# ---------------------------------------------------------------------------
# RETRIEVAL PARAMETERS
# ---------------------------------------------------------------------------

TOP_K_RETRIEVAL: int = 10
"""
Number of candidates retrieved from both FAISS and BM25 before fusion.

Retrieve broadly for recall (10), rerank precisely for precision (TOP_K_RERANK=3).
Higher values increase recall but add cross-encoder latency linearly.
"""

TOP_K_RERANK: int = 3
"""
Number of chunks passed to the LLM after cross-encoder reranking.

3 chunks × ~500 words each ≈ 1500 words of context, well within Gemini's 1M
context window. Enough to answer most focused document questions.
"""

RRF_CONSTANT: int = 60
"""
Reciprocal Rank Fusion constant (k).

RRF score formula: sum(1 / (rank + k)) across retrieval systems.

k=60 is the standard value from the original paper:
  Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion outperforms
  Condorcet and individual rank learning methods." SIGIR 2009.

Higher k dampens the influence of top-ranked results; lower k amplifies it.
k=60 provides robust performance across query types without per-dataset tuning.
"""

SIMILARITY_THRESHOLD_GEMINI: float = 0.5
"""
Minimum FAISS cosine similarity score to proceed with retrieval (Gemini embeddings).

If the top FAISS result scores below this, we refuse without calling the LLM.
This is Anti-Hallucination Layer 1 — see architecture overview in app.py.

0.5 calibrated against Gemini text-embedding-004 score distribution:
  < 0.3 = unrelated
  0.3-0.5 = loosely related (often still off-topic)
  > 0.5 = genuinely relevant

This is a tunable hyperparameter. Increase to reduce false-positives (tighter
refusal), decrease to allow more borderline retrievals through.
"""

SIMILARITY_THRESHOLD_FALLBACK: float = 0.35
"""
Minimum FAISS cosine similarity for the sentence-transformers fallback model.

Lower than GEMINI threshold because all-MiniLM-L6-v2 produces a different score
distribution — its vectors are lower-dimensional (384 vs 768) and tend to produce
lower absolute cosine scores for equivalent semantic similarity.

Using the same threshold across both models would result in either:
  - Too many false positives with fallback (threshold too low for Gemini)
  - Too many false negatives with Gemini (threshold too high for fallback)
"""


# ---------------------------------------------------------------------------
# CONVERSATION MEMORY
# ---------------------------------------------------------------------------

MAX_HISTORY_MESSAGES: int = 10
"""
Maximum number of messages (user + assistant turns) kept in the active context window.

Each message ≈ 200-300 tokens on average. 10 messages ≈ 2000-3000 tokens,
leaving plenty of headroom in Gemini's 1M context for the system prompt + chunks.

Truncation strategy: keep the MOST RECENT messages (sliding window, not summary).
v1 plan: Replace sliding window with summarisation of older turns.
"""


# ---------------------------------------------------------------------------
# REDIS TTL (Time-To-Live) VALUES
# ---------------------------------------------------------------------------

REDIS_SESSION_TTL: int = 3600
"""
Session history TTL in seconds (1 hour).

TTL is reset on every message write, so active sessions never expire mid-conversation.
Sessions that go idle for >1 hour are automatically cleaned up by Redis.
"""

REDIS_INDEX_TTL: int = 86400
"""
PDF index cache TTL in seconds (24 hours).

FAISS index, BM25 index, and chunk metadata are stored under this TTL.
After 24 hours, the next upload of the same PDF triggers re-ingestion.
Balances storage cost (Redis free tier: 30MB) vs re-ingestion latency.
"""


# ---------------------------------------------------------------------------
# HEADER/FOOTER FILTER PARAMETERS (used in core/parser.py)
# ---------------------------------------------------------------------------

HEADER_FOOTER_FREQ_THRESHOLD: float = 0.6
"""
Minimum frequency (fraction of pages) for a line to be classified as a
header/footer and stripped.

0.6 = appears on 60%+ of pages → likely a repeating header/footer.
Only applied to short lines (< 8 words) to avoid stripping actual content.
"""

HEADER_FOOTER_MIN_PAGES: int = 5
"""
Minimum number of pages required before the header/footer filter activates.

PDFs with fewer than 5 pages are unlikely to have meaningful repeating headers,
and applying frequency-based filtering on small page counts produces false positives.
"""


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

LOG_FILE: str = "agent.log"
"""
Output path for the file-based log handler. Written to the project root.
Evaluators can inspect this with: grep "[RETRIEVER]" agent.log
"""

LOG_LEVEL: str = "INFO"
"""
Logging verbosity level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.
Set to DEBUG for development, INFO for production/evaluation runs.
"""

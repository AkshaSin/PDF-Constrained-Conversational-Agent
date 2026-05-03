"""
utils/logger.py — Structured Logging Module
=============================================

Provides a single `get_logger(name)` factory used by every other module.
All loggers share one root handler configured here at import time, so:
  - You never duplicate handler setup.
  - Log level changes in one place propagate everywhere.
  - Optional file logging is a single flag in config.py.

Design references
-----------------
* Python logging HOWTO: https://docs.python.org/3/howto/logging.html
* 12-Factor App (factor XI — Logs): https://12factor.net/logs
  The 12-factor principle says an app should treat logs as event streams
  written to stdout. We honour this: console is always on; file is optional.

Usage
-----
    from utils.logger import get_logger

    log = get_logger(__name__)
    log.info("PDF ingested", extra={"pages": 42})
    log.warning("Low similarity score: %.3f", score)
    log.error("Redis connection failed: %s", err)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colour codes — makes level names pop in the terminal.
# Falls back gracefully on Windows terminals that don't support ANSI
# (the ColourFormatter strips codes when stdout is not a TTY).
# ---------------------------------------------------------------------------
_COLOURS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[41m",   # Red background
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    """
    A logging.Formatter subclass that adds ANSI colour to the level name
    **only** when the output stream is an interactive terminal (TTY).

    Why TTY-check?
        When logs are piped to a file or another process, ANSI escape codes
        become noise (e.g. `cat log.txt` shows garbled ^[[32m sequences).
        `stream.isatty()` returns False for redirected streams, so we
        automatically fall back to plain text in that case.
    """

    def __init__(self, fmt: str, datefmt: str, stream: object) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_colour: bool = hasattr(stream, "isatty") and stream.isatty()

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        formatted = super().format(record)
        if self._use_colour:
            colour = _COLOURS.get(record.levelname, "")
            # Only colour the level-name token for readability
            formatted = formatted.replace(
                record.levelname,
                f"{colour}{record.levelname}{_RESET}",
                1,  # replace first occurrence only
            )
        return formatted


# ---------------------------------------------------------------------------
# Root logger setup — runs ONCE when this module is first imported.
# Every subsequent call to get_logger() returns a child of "pdf_agent".
# ---------------------------------------------------------------------------
_ROOT_LOGGER_NAME = "pdf_agent"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_root_configured = False  # guard: idempotent even if module reloaded


def _configure_root_logger(
    level: int = logging.DEBUG,
    log_file: Optional[str] = None,
) -> None:
    """
    Attach handlers to the 'pdf_agent' root logger.

    Called automatically by get_logger() the first time it runs.
    Separated into its own function so tests can call it explicitly with
    different parameters without touching the module-level guard.

    Args:
        level:    Minimum log level (e.g. logging.INFO). DEBUG by default
                  so all messages are captured; narrow in production via
                  config.py's LOG_LEVEL constant.
        log_file: Optional path to a .log file. If provided, all records
                  are also written there (no colour, plain text).
    """
    global _root_configured
    if _root_configured:
        return

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)

    # --- Console handler (always present) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        _ColourFormatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT, stream=sys.stdout)
    )
    root.addHandler(console_handler)

    # --- File handler (optional) ---
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        )
        root.addHandler(file_handler)

    # Prevent log records from propagating to the root Python logger
    # (which would double-print everything if basicConfig was called elsewhere)
    root.propagate = False

    _root_configured = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger of the 'pdf_agent' root logger.

    Recommended usage — pass __name__ so the logger name maps 1-to-1
    with the module file, making logs trivially traceable:

        log = get_logger(__name__)

    The first call to this function initialises the root logger (lazy init).
    The log level and optional file path are read from config.py if available,
    falling back to DEBUG-level console-only logging if config isn't ready yet.

    Args:
        name: Logger name — use `__name__` (the module's dotted path).

    Returns:
        A configured logging.Logger instance.
    """
    # Lazy import of config to avoid circular imports at module load time.
    # config.py imports nothing from utils/, so the cycle won't occur, but
    # deferred import is defensive hygiene.
    try:
        import config  # noqa: PLC0415
        level_str: str = getattr(config, "LOG_LEVEL", "DEBUG").upper()
        level: int = getattr(logging, level_str, logging.DEBUG)
        log_file: Optional[str] = getattr(config, "LOG_FILE", None)
    except (ImportError, AttributeError):
        level = logging.DEBUG
        log_file = None

    _configure_root_logger(level=level, log_file=log_file)

    # Child logger name: "pdf_agent.<caller_module>"
    # e.g. get_logger("ingestion.chunker") → "pdf_agent.ingestion.chunker"
    child_name = f"{_ROOT_LOGGER_NAME}.{name}" if not name.startswith(_ROOT_LOGGER_NAME) else name
    return logging.getLogger(child_name)

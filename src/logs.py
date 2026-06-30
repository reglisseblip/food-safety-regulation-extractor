"""
logs — logging configuration using Rich (colored, structured output).

Call `setup()` once at startup (in a `__main__` or the app).
Each module simply does `log = logging.getLogger(__name__)` and `log.info(...)`.
"""
from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup(level: int = logging.INFO) -> None:
    """Enable Rich logs (color, level, timestamp). Use `level=logging.DEBUG` to see every LLM call."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
    )
    # Silence third-party HTTP noise while keeping our own logs.
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

"""Loading and cleaning the raw list of names used to train the model."""

from __future__ import annotations

from pathlib import Path
from typing import List


def load_words(path: str | Path) -> List[str]:
    """Read a newline-delimited text file of names and clean it up.

    Each line is stripped of surrounding whitespace and lower-cased.
    Empty lines are dropped.

    Parameters
    ----------
    path:
        Path to a plain-text file containing one name per line.

    Returns
    -------
    list[str]
        The cleaned list of names.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    words = [line.strip().lower() for line in raw_lines if line.strip()]

    if not words:
        raise ValueError(f"No names found in {path!s}")

    return words


def describe_words(words: List[str]) -> dict:
    """Return basic summary statistics about a list of names.

    Mirrors the quick sanity-check printed in the original notebook.
    """
    return {
        "count": len(words),
        "first_10": words[:10],
        "shortest": min(words, key=len),
        "longest": max(words, key=len),
        "avg_length": sum(len(w) for w in words) / len(words),
    }

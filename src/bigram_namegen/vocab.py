"""Character vocabulary and integer<->character lookup tables.

A special start/end-of-word token, ``'.'``, is always placed at index 0.
It marks both the beginning and the end of a name, which lets the bigram
model learn "what character tends to start a name" and "what character
tends to end a name" using the exact same machinery it uses for every
other character transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

START_END_TOKEN = "."


@dataclass(frozen=True)
class Vocab:
    """Container bundling the character set with its lookup tables."""

    chars: List[str]
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @property
    def size(self) -> int:
        return len(self.chars)


def build_vocab(words: List[str]) -> Vocab:
    """Build the character vocabulary from a list of names.

    The vocabulary is the sorted set of every character that appears in
    ``words``, with the ``'.'`` token prepended at index 0.
    """
    chars = sorted(set("".join(words)))
    chars = [START_END_TOKEN] + chars

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    return Vocab(chars=chars, stoi=stoi, itos=itos)


def iter_bigrams(word: str):
    """Yield the (ch1, ch2) bigrams in a word, including the wrapper token.

    Example
    -------
    >>> list(iter_bigrams("ada"))
    [('.', 'a'), ('a', 'd'), ('d', 'a'), ('a', '.')]
    """
    chars = [START_END_TOKEN] + list(word) + [START_END_TOKEN]
    yield from zip(chars, chars[1:])

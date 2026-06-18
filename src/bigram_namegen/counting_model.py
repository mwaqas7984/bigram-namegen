"""The baseline statistical model: count bigrams, normalize into probabilities.

This is the simplest possible language model for this task. It has no
learned parameters in the gradient-descent sense; it just counts how often
each character follows each other character in the training data, and
turns those counts into a probability distribution.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from .vocab import Vocab, iter_bigrams


def build_bigram_counts(words: List[str], vocab: Vocab) -> torch.Tensor:
    """Build the ``(vocab_size, vocab_size)`` bigram count matrix ``N``.

    ``N[i, j]`` is the number of times character ``j`` directly followed
    character ``i`` anywhere in the training data (including the
    start/end token).
    """
    N = torch.zeros((vocab.size, vocab.size), dtype=torch.int32)

    for word in words:
        for ch1, ch2 in iter_bigrams(word):
            ix1, ix2 = vocab.stoi[ch1], vocab.stoi[ch2]
            N[ix1, ix2] += 1

    return N


def counts_to_probabilities(N: torch.Tensor) -> torch.Tensor:
    """Convert raw counts into a row-normalized probability matrix.

    Laplace ("add-one") smoothing is applied first so that no transition
    ever has exactly zero probability — otherwise a single unseen bigram
    at evaluation time would produce an infinite loss.

    .. math::
        P[i, j] = \\frac{N[i, j] + 1}{\\sum_k (N[i, k] + 1)}
    """
    P = (N + 1).float()
    P = P / P.sum(dim=1, keepdim=True)
    return P


def generate_names_counting(
    P: torch.Tensor,
    vocab: Vocab,
    num_names: int = 10,
    seed: int = 2147483647,
) -> List[str]:
    """Sample new names from the counting model's probability table.

    Starting from the start token, repeatedly sample the next character
    from ``P[current_index]`` until the start/end token is sampled again.
    """
    generator = torch.Generator().manual_seed(seed)
    names = []

    for _ in range(num_names):
        name_chars: List[str] = []
        ix = 0  # '.' token

        while True:
            row = P[ix]
            ix = torch.multinomial(
                row, num_samples=1, replacement=True, generator=generator
            ).item()

            if ix == 0:
                break

            name_chars.append(vocab.itos[ix])

        names.append("".join(name_chars))

    return names


def nll_loss_counting(words: List[str], P: torch.Tensor, vocab: Vocab) -> Tuple[float, int]:
    """Compute the average negative log-likelihood of the data under ``P``.

    Lower is better. This is the quantity the neural model is trained to
    minimize, so it doubles as a baseline to compare the neural model
    against.

    Returns
    -------
    (average_nll, num_bigrams)
    """
    log_likelihood = 0.0
    n = 0

    for word in words:
        for ch1, ch2 in iter_bigrams(word):
            ix1, ix2 = vocab.stoi[ch1], vocab.stoi[ch2]
            prob = P[ix1, ix2]
            log_likelihood += torch.log(prob).item()
            n += 1

    avg_nll = -log_likelihood / n
    return avg_nll, n

"""A single-layer neural network trained by gradient descent.

This network is intentionally minimal: it is mathematically equivalent to
multinomial logistic regression over one-hot encoded characters. The point
is pedagogical — to show that the same bigram statistics learned by simple
counting (see ``counting_model.py``) can also be *learned* via gradient
descent, and that the two approaches converge to the same loss.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from .vocab import Vocab, iter_bigrams


def build_training_pairs(words: List[str], vocab: Vocab) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn the names into (input, target) integer-index tensors.

    For every bigram ``(ch1, ch2)`` in the dataset, ``ch1``'s index goes
    into ``xs`` and ``ch2``'s index goes into the parallel ``ys``.
    """
    xs, ys = [], []

    for word in words:
        for ch1, ch2 in iter_bigrams(word):
            xs.append(vocab.stoi[ch1])
            ys.append(vocab.stoi[ch2])

    return torch.tensor(xs), torch.tensor(ys)


def init_weights(vocab: Vocab, seed: int = 2147483647) -> torch.Tensor:
    """Randomly initialize the ``(vocab_size, vocab_size)`` weight matrix."""
    generator = torch.Generator().manual_seed(seed)
    W = torch.randn((vocab.size, vocab.size), generator=generator, requires_grad=True)
    return W


def train_neural_model(
    words: List[str],
    vocab: Vocab,
    steps: int = 200,
    lr: float = 50.0,
    seed: int = 2147483647,
    log_every: int = 20,
    verbose: bool = True,
) -> Tuple[torch.Tensor, List[float]]:
    """Train the single-layer network with vanilla gradient descent.

    Forward pass, per step:
        1. one-hot encode the inputs -> ``xenc``
        2. ``logits = xenc @ W``
        3. ``counts = exp(logits)``                  (always positive)
        4. ``probs = counts / counts.sum(dim=1)``     (softmax, row-normalized)
        5. loss = mean negative log-probability of the correct target

    Returns
    -------
    (W, loss_history)
    """
    xs, ys = build_training_pairs(words, vocab)
    W = init_weights(vocab, seed=seed)
    loss_history: List[float] = []

    for step in range(steps):
        xenc = F.one_hot(xs, num_classes=vocab.size).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True)

        correct_probs = probs[torch.arange(xs.shape[0]), ys]
        loss = -correct_probs.log().mean()
        loss_history.append(loss.item())

        W.grad = None
        loss.backward()
        W.data -= lr * W.grad

        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:>4} | loss: {loss.item():.4f}")

    return W, loss_history


def generate_names_neural(
    W: torch.Tensor,
    vocab: Vocab,
    num_names: int = 20,
    seed: int = 2147483647,
) -> List[str]:
    """Sample new names using the trained weight matrix ``W``."""
    generator = torch.Generator().manual_seed(seed)
    names = []

    for _ in range(num_names):
        name_chars: List[str] = []
        ix = 0

        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab.size).float()
            logits = xenc @ W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)

            ix = torch.multinomial(
                probs, num_samples=1, replacement=True, generator=generator
            ).item()

            if ix == 0:
                break

            name_chars.append(vocab.itos[ix])

        names.append("".join(name_chars))

    return names

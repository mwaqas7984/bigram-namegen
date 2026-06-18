"""
bigram_namegen
==============

A small, from-scratch character-level bigram language model for generating
new names (e.g. fantasy villain names), implemented two ways:

1. A pure **counting** model — count bigram frequencies and turn them into
   a probability table (with Laplace smoothing).
2. A **neural** model — a single-layer "network" (effectively logistic
   regression over one-hot encoded characters) trained with gradient
   descent to do the same job, optimized via negative log-likelihood.

This package is a refactor of an exploratory Jupyter notebook into a
reusable, testable, command-line-friendly library.
"""

from .vocab import Vocab, build_vocab
from .data import load_words
from .counting_model import (
    build_bigram_counts,
    counts_to_probabilities,
    generate_names_counting,
    nll_loss_counting,
)
from .neural_model import (
    init_weights,
    train_neural_model,
    generate_names_neural,
)

__version__ = "0.1.0"

__all__ = [
    "Vocab",
    "build_vocab",
    "load_words",
    "build_bigram_counts",
    "counts_to_probabilities",
    "generate_names_counting",
    "nll_loss_counting",
    "init_weights",
    "train_neural_model",
    "generate_names_neural",
]

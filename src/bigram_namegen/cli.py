"""Command-line interface for training and sampling from the bigram models.

Examples
--------
Train both models on the bundled villain-names dataset and generate samples::

    python -m bigram_namegen --data data/villain_names.txt --model both --num-names 20

Train only the neural model for longer, with a custom learning rate::

    python -m bigram_namegen --data data/villain_names.txt --model neural \\
        --steps 500 --lr 20 --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .counting_model import (
    build_bigram_counts,
    counts_to_probabilities,
    generate_names_counting,
    nll_loss_counting,
)
from .data import describe_words, load_words
from .neural_model import generate_names_neural, train_neural_model
from .plotting import plot_loss_curve
from .vocab import build_vocab


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bigram_namegen",
        description="Train a character-level bigram model and generate new names.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/villain_names.txt",
        help="Path to a newline-delimited text file of training names.",
    )
    parser.add_argument(
        "--model",
        choices=["counting", "neural", "both"],
        default="both",
        help="Which model(s) to run.",
    )
    parser.add_argument("--num-names", type=int, default=20, help="Names to sample per model.")
    parser.add_argument("--steps", type=int, default=200, help="Neural model training steps.")
    parser.add_argument("--lr", type=float, default=50.0, help="Neural model learning rate.")
    parser.add_argument("--seed", type=int, default=2147483647, help="Random seed.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a PNG of the neural model's loss curve (requires --model neural/both).",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="nn_loss.png",
        help="Where to save the loss-curve PNG.",
    )
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    words = load_words(args.data)
    stats = describe_words(words)
    print(f"Loaded {stats['count']} names from {args.data}")
    print(f"  shortest={stats['shortest']!r} longest={stats['longest']!r} "
          f"avg_len={stats['avg_length']:.1f}")

    vocab = build_vocab(words)
    print(f"Vocabulary size: {vocab.size} ({''.join(vocab.chars)!r})\n")

    baseline_nll = None

    if args.model in ("counting", "both"):
        print("=== Counting model ===")
        N = build_bigram_counts(words, vocab)
        P = counts_to_probabilities(N)
        baseline_nll, n_bigrams = nll_loss_counting(words, P, vocab)
        print(f"Average NLL over {n_bigrams} bigrams: {baseline_nll:.4f}")

        names = generate_names_counting(P, vocab, num_names=args.num_names, seed=args.seed)
        print("Generated names:")
        for name in names:
            print(f"  {name}")
        print()

    if args.model in ("neural", "both"):
        print("=== Neural model ===")
        W, loss_history = train_neural_model(
            words, vocab, steps=args.steps, lr=args.lr, seed=args.seed
        )

        if baseline_nll is not None:
            print(f"\n(For comparison, counting-model loss: {baseline_nll:.4f})")

        names = generate_names_neural(W, vocab, num_names=args.num_names, seed=args.seed)
        print("Generated names:")
        for name in names:
            print(f"  {name}")

        if args.plot:
            out = plot_loss_curve(loss_history, baseline=baseline_nll, output_path=args.plot_output)
            print(f"\nSaved loss curve to {out}")


if __name__ == "__main__":
    main()

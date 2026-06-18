import torch

from bigram_namegen.counting_model import (
    build_bigram_counts,
    counts_to_probabilities,
    generate_names_counting,
    nll_loss_counting,
)
from bigram_namegen.vocab import build_vocab

WORDS = ["ab", "ab", "ac"]


def test_build_bigram_counts_matches_manual_count():
    vocab = build_vocab(WORDS)
    N = build_bigram_counts(WORDS, vocab)

    # '.' -> 'a' should happen 3 times (once per word)
    assert N[vocab.stoi["."], vocab.stoi["a"]].item() == 3
    # 'a' -> 'b' should happen 2 times, 'a' -> 'c' once
    assert N[vocab.stoi["a"], vocab.stoi["b"]].item() == 2
    assert N[vocab.stoi["a"], vocab.stoi["c"]].item() == 1


def test_probabilities_rows_sum_to_one():
    vocab = build_vocab(WORDS)
    N = build_bigram_counts(WORDS, vocab)
    P = counts_to_probabilities(N)

    row_sums = P.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_generate_names_counting_is_deterministic_with_seed():
    vocab = build_vocab(WORDS)
    N = build_bigram_counts(WORDS, vocab)
    P = counts_to_probabilities(N)

    names_a = generate_names_counting(P, vocab, num_names=5, seed=42)
    names_b = generate_names_counting(P, vocab, num_names=5, seed=42)
    assert names_a == names_b


def test_nll_loss_is_finite_and_positive():
    vocab = build_vocab(WORDS)
    N = build_bigram_counts(WORDS, vocab)
    P = counts_to_probabilities(N)

    avg_nll, n_bigrams = nll_loss_counting(WORDS, P, vocab)
    assert n_bigrams == sum(len(w) + 1 for w in WORDS)
    assert avg_nll > 0
    assert avg_nll == avg_nll  # not NaN

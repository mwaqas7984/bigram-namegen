from bigram_namegen.neural_model import (
    build_training_pairs,
    generate_names_neural,
    init_weights,
    train_neural_model,
)
from bigram_namegen.vocab import build_vocab

WORDS = ["ab", "ab", "ac", "ad"]


def test_build_training_pairs_lengths_match_bigram_count():
    vocab = build_vocab(WORDS)
    xs, ys = build_training_pairs(WORDS, vocab)
    expected = sum(len(w) + 1 for w in WORDS)
    assert xs.shape[0] == expected
    assert ys.shape[0] == expected


def test_init_weights_shape_and_requires_grad():
    vocab = build_vocab(WORDS)
    W = init_weights(vocab, seed=0)
    assert W.shape == (vocab.size, vocab.size)
    assert W.requires_grad


def test_training_reduces_loss():
    vocab = build_vocab(WORDS)
    _, loss_history = train_neural_model(WORDS, vocab, steps=50, lr=10.0, verbose=False)
    assert loss_history[-1] < loss_history[0]


def test_generate_names_neural_is_deterministic_with_seed():
    vocab = build_vocab(WORDS)
    W, _ = train_neural_model(WORDS, vocab, steps=10, lr=10.0, verbose=False)

    names_a = generate_names_neural(W, vocab, num_names=5, seed=7)
    names_b = generate_names_neural(W, vocab, num_names=5, seed=7)
    assert names_a == names_b

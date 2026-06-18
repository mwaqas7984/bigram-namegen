from bigram_namegen.vocab import build_vocab, iter_bigrams
from bigram_namegen.data import describe_words


def test_iter_bigrams_wraps_with_start_end_token():
    assert list(iter_bigrams("ada")) == [
        (".", "a"),
        ("a", "d"),
        ("d", "a"),
        ("a", "."),
    ]


def test_build_vocab_includes_token_and_is_sorted():
    vocab = build_vocab(["bob", "ada"])
    assert vocab.chars[0] == "."
    assert vocab.chars == [".", "a", "b", "d", "o"]
    assert vocab.stoi["."] == 0
    assert vocab.itos[0] == "."
    assert vocab.size == len(vocab.chars)


def test_describe_words_basic_stats():
    stats = describe_words(["ab", "abc", "a"])
    assert stats["count"] == 3
    assert stats["shortest"] == "a"
    assert stats["longest"] == "abc"
    assert stats["avg_length"] == (2 + 3 + 1) / 3

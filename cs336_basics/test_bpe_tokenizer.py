import pytest

from cs336_basics.bpe_tokenizer import BpeTokenizer


def test_bpe_pretokenization_single_process():
    """Single process pretokenization test."""
    tokenizer = BpeTokenizer(
        vocab_size=100,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        pretokenization_num_processes=1,
    )
    pretoken_counts = tokenizer.pretokenize(
        "cs336_basics/test_data/pretokenizer_test.txt"
    )
    assert isinstance(pretoken_counts, dict)

    expected_pretoken_counts = {
        b" ": 3,
        b"\n": 2,
        b"test": 1,
        b"hello": 1,
        b",": 1,
        b" another": 1,
        b" test": 1,
        b" \xe6\xb5\x8b\xe8\xaf\x95": 1,
        b" \xe4\xb8\xad\xe6\x96\x87": 1,
        b"\xe5\x8f\xa6\xe5\xa4\x96\xe4\xb8\x80\xe4\xb8\xaatest": 1,
    }
    assert pretoken_counts == expected_pretoken_counts


def test_bpe_pretokenization_multi_process():
    """Multi-process pretokenization test."""
    tokenizer = BpeTokenizer(
        vocab_size=100,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        pretokenization_num_processes=4,
    )
    pretoken_counts = tokenizer.pretokenize(
        "cs336_basics/test_data/pretokenizer_test.txt"
    )
    assert isinstance(pretoken_counts, dict)

    expected_pretoken_counts = {
        b" ": 3,
        b"\n": 2,
        b"test": 1,
        b"hello": 1,
        b",": 1,
        b" another": 1,
        b" test": 1,
        b" \xe6\xb5\x8b\xe8\xaf\x95": 1,
        b" \xe4\xb8\xad\xe6\x96\x87": 1,
        b"\xe5\x8f\xa6\xe5\xa4\x96\xe4\xb8\x80\xe4\xb8\xaatest": 1,
    }
    assert pretoken_counts == expected_pretoken_counts


def test_tokenizer_train_single_process():
    """Test training the BPE tokenizer using a single process."""
    tokenizer = BpeTokenizer(
        vocab_size=(2 + 256 + 6),
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        pretokenization_num_processes=1,
    )
    vocab, merges = tokenizer.train("cs336_basics/test_data/tokenizer_train_test.txt")
    expected_merges = [
        (b"s", b"t"),
        (b"e", b"st"),
        (b"o", b"w"),
        (b"l", b"ow"),
        (b"w", b"est"),
        (b"n", b"e"),
    ]

    assert isinstance(vocab, dict)
    assert isinstance(merges, list)
    assert len(vocab) == (2 + 256 + 6)  # special tokens + byte vocab + 6 merges
    assert all(token in vocab.values() for token in [b"<|endoftext|>", b"<|pad|>"])
    assert merges == expected_merges

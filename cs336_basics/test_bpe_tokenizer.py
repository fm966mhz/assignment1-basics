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


def test_encode_one_pretoken():
    """Test encoding a single pretoken."""
    tokenizer = BpeTokenizer(
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        vocab={
            0: b"<|endoftext|>",
            1: b"<|pad|>",
            2: b"t",
            3: b"h",
            4: b"e",
            5: b"th",
            6: b"he",
            7: b"the",
        },
        merges=[(b"h", b"e"), (b"t", b"h"), (b"th", b"e"), (b"t", b"he")],
    )

    token_ids = tokenizer.encode("the")
    expected_token_ids = [7]  # "the" is a single token in the vocab
    assert token_ids == expected_token_ids


def test_encode_one_pretoken_repeated_bytes():
    """Test encoding a single pretoken."""
    tokenizer = BpeTokenizer(
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        vocab={
            0: b"<|endoftext|>",
            1: b"<|pad|>",
            2: b"l",
            3: b"ll",
            4: b"lll",
            5: b"llll",
        },
        merges=[(b"l", b"l"), (b"ll", b"l"), (b"lll", b"l"), (b"l", b"ll")],
    )
    assert tokenizer.encode("llll") == [3, 3]

    tokenizer = BpeTokenizer(
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        vocab={
            0: b"<|endoftext|>",
            1: b"<|pad|>",
            2: b"l",
            3: b"ll",
            4: b"lll",
            5: b"llll",
        },
        merges=[
            (b"l", b"l"),
            (b"ll", b"l"),
            (b"lll", b"l"),
            (b"l", b"ll"),
            (b"ll", b"ll"),
        ],
    )
    assert tokenizer.encode("llll") == [5]  # "llll" is a single token in the vocab

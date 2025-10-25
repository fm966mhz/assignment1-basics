"""Byte Pair Encoding (BPE) tokenizer implementation."""

import pickle
import heapq

from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Iterable, Iterator

import regex as re

from .pretokenization_example import find_chunk_boundaries


_DEFAULT_PRETOKENIZATION_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
_SPLIT_SPECIAL_TOKEN = "<|endoftext|>"


@dataclass(frozen=True)
class BytesPair:
    """A pair of bytes used in BPE merges.

    This is mostly needed because the solution wants to break ties by taking the max of the bytes
    pairs of the same count. `heapq` in Python is a min-heap, so we need to create a custom class
    and override the less-than operator to achieve this.
    """

    first: bytes
    second: bytes

    def __lt__(self, other: "BytesPair") -> bool:
        """Less-than operator for BytesPair.

        We want to break ties by taking the max of the bytes pairs, so we invert the comparison.
        """
        if self.first != other.first:
            return self.first > other.first
        return self.second > other.second


@dataclass
class BytesPairListNode:
    """A node in the bytes pair linked list."""

    bytes_pair: BytesPair
    prev: "BytesPairListNode | None" = None
    next: "BytesPairListNode | None" = None


@dataclass
class PretokenInfo:
    """Information about a pretoken in the BPE training process."""

    count: int
    first_node: BytesPairListNode | None = None


class BpeTokenizer:
    """A Byte Pair Encoding tokenizer implementation."""

    def __init__(
        self,
        *,
        vocab_size: int | None = None,
        pretokenization_regex: str = _DEFAULT_PRETOKENIZATION_REGEX,
        split_special_token: str | None = _SPLIT_SPECIAL_TOKEN,
        special_tokens: list[str] | None = None,
        pretokenization_num_processes: int = 4,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        """Initialize the BPE tokenizer.
        Args:
            vocab_size (int): The desired vocabulary size.
            pretokenization_regex (str): The regex pattern for pretokenization.
            split_special_token (str | None): Special token used to split the input during
                pretokenization.
            special_tokens (list[str] | None): List of special tokens to include in the vocabulary.
            pretokenization_num_processes (int): Number of processes to use for pretokenization.
            vocab (dict[int, bytes] | None): Predefined vocabulary to use. If provided, training
                related arguments will be ignored, such as vocab_size.
            merges (list[tuple[bytes, bytes]] | None): Predefined merges to use. If provided,
                training related arguments will be ignored, such as vocab_size.
        """
        # Compile the pretokenization regex pattern.
        self._pretokenization_regex_pattern = re.compile(
            pretokenization_regex, re.UNICODE
        )
        # Handle special tokens.
        self._split_special_token = split_special_token or _SPLIT_SPECIAL_TOKEN
        self._special_tokens = set(special_tokens or [])
        if self._split_special_token not in self._special_tokens:
            self._special_tokens.add(self._split_special_token)
        self._special_tokens_split_regex = "|".join(
            re.escape(token) for token in self._special_tokens
        )
        # Number of processes to use for pretokenization.
        self._pretokenization_num_processes = pretokenization_num_processes
        # Trained vocab and merges will be stored here after training.
        if vocab is not None and merges is not None:
            self._vocab = vocab
            self._merges = [
                BytesPair(first=merge[0], second=merge[1]) for merge in merges
            ]
            self._vocab_size = len(vocab)
        elif vocab is None and merges is None and vocab_size is not None:
            self._vocab: dict[int, bytes] = {
                i: special_token.encode("utf-8")
                for i, special_token in enumerate(sorted(self._special_tokens))
            }
            for i in range(256):
                self._vocab[i + len(self._special_tokens)] = bytes([i])
            self._merges: list[BytesPair] = []
            self._vocab_size = vocab_size
        else:
            raise ValueError(
                "Both vocab and merges must be provided together, or neither."
            )
        # Data structures needed for encoding.
        self._merges_to_idx: dict[BytesPair, int] = {}
        self._inverted_vocab: dict[bytes, int] = {}
        # Longest vocab token length in utf-8 characters. This is useful for encoding.
        self._longest_vocab_length: int = 0

    def train(
        self,
        input_path: str,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train the BPE tokenizer on the given input file.

        Args:
            input_path (str): Path to the input text file.
        """
        pretoken_counts = self.pretokenize(input_path)
        pretoken_info_list, bytes_pair_to_pretoken_positions, bytes_pair_counts = (
            self._preprocess_pretoken_counts(pretoken_counts)
        )
        # Build the initial heap of byte pairs based on their counts.
        heap: list[tuple[int, BytesPair]] = [
            (-count, bytes_pair) for bytes_pair, count in bytes_pair_counts.items()
        ]
        heapq.heapify(heap)
        # Perform BPE merges until reaching the desired vocab size.
        while len(self._vocab) < self._vocab_size and heap:
            neg_count, most_frequent_pair = heapq.heappop(heap)
            current_count = bytes_pair_counts.get(most_frequent_pair, 0)
            if current_count == 0:
                continue
            # If the count has changed, reinsert with the updated count.
            if -neg_count != current_count:
                heapq.heappush(heap, (-current_count, most_frequent_pair))
                continue
            # Perform the merge.
            new_token = most_frequent_pair.first + most_frequent_pair.second
            self._vocab[len(self._vocab)] = new_token
            self._merges.append(most_frequent_pair)

            # Update pretokens containing the merged pair.
            self._bpe_merge(
                most_frequent_pair,
                bytes_pair_to_pretoken_positions,
                bytes_pair_counts,
                heap,
                pretoken_info_list,
            )
        return self._vocab, self.get_merge_as_list_of_tuples()

    def _pretokenize_one_chunk(
        self,
        input_path: str,
        start: int,
        end: int,
    ) -> Counter[bytes]:
        """
        Pretokenize the input text into a list of byte strings.

        Args:
            input_path (str): Path to the input text file.
            start (int): Start byte position of the chunk.
            end (int): End byte position of the chunk.

        Returns:
            Counter[bytes]: Counter of byte strings representing the pretokenized text.
        """
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk_size = end - start
            chunk_bytes = f.read(chunk_size)
            chunk_string = chunk_bytes.decode("utf-8", errors="ignore")
        pretoken_counts: Counter[bytes] = Counter()
        for split_part in re.split(self._special_tokens_split_regex, chunk_string):
            if not split_part:
                continue
            for match in self._pretokenization_regex_pattern.finditer(split_part):
                pretoken = match.group(0).encode("utf-8")
                pretoken_counts[pretoken] += 1
        return pretoken_counts

    def pretokenize(
        self,
        input_path: str,
    ) -> Counter[bytes]:
        """
        Pretokenize the input file into a list of byte strings.

        Args:
            input_path (str): Path to the input text file.

        Returns:
            Counter[bytes]: Counter of byte strings representing the pretokenized text.
        """
        pretoken_counts: Counter[bytes] = Counter()

        with open(input_path, "rb") as f:
            chunk_boundaries = find_chunk_boundaries(
                file=f,
                desired_num_chunks=self._pretokenization_num_processes,
                split_special_token=self._split_special_token.encode("utf-8"),
            )

        with Pool(processes=self._pretokenization_num_processes) as pool:
            results = []
            for bi in range(len(chunk_boundaries) - 1):
                start = chunk_boundaries[bi]
                end = chunk_boundaries[bi + 1]
                results.append(
                    pool.apply_async(
                        self._pretokenize_one_chunk,
                        args=(input_path, start, end),
                    )
                )
            for r in results:
                pretoken_counts += r.get()

        return pretoken_counts

    def _preprocess_pretoken_counts(
        self,
        pretoken_counts: Counter[bytes],
    ) -> tuple[
        list[PretokenInfo],
        dict[BytesPair, set[int]],
        dict[BytesPair, int],
    ]:
        """
        Preprocess the pretoken counts into a structure suitable for BPE training.

        Args:
            pretoken_counts (Counter[bytes]): Counter of byte strings representing the pretokenized
                text.

        Returns:
            list[PretokenInfo]: List of PretokenInfo objects for BPE training.
            dict[tuple[bytes, bytes], set[int]]: inverted list from byte pairs to the positions of
                the pretokens that generate such pairs.
            dict[tuple[bytes, bytes], int]: counts of each byte pair across all pretokens.
        """
        pretoken_info_list: list[PretokenInfo] = []
        bytes_pair_to_pretoken_positions: dict[BytesPair, set[int]] = defaultdict(set)
        bytes_pair_counts: dict[BytesPair, int] = defaultdict(int)

        for pretoken, count in pretoken_counts.items():
            # Build the linked list of byte pairs for this pretoken
            first_node: BytesPairListNode | None = None
            prev_node: BytesPairListNode | None = None
            byte_sequence = list(pretoken)
            for i in range(len(byte_sequence) - 1):
                bytes_pair = BytesPair(
                    first=bytes([byte_sequence[i]]),
                    second=bytes([byte_sequence[i + 1]]),
                )
                current_node = BytesPairListNode(bytes_pair=bytes_pair)
                bytes_pair_counts[bytes_pair] += count
                if first_node is None:
                    first_node = current_node
                if prev_node is not None:
                    prev_node.next = current_node
                    current_node.prev = prev_node
                prev_node = current_node

                # Update the inverted index
                bytes_pair_to_pretoken_positions[bytes_pair].add(
                    len(pretoken_info_list)
                )

            pretoken_info = PretokenInfo(count=count, first_node=first_node)
            pretoken_info_list.append(pretoken_info)

        return pretoken_info_list, bytes_pair_to_pretoken_positions, bytes_pair_counts

    def _bpe_merge(
        self,
        most_frequent_pair: BytesPair,
        bytes_pair_to_pretoken_positions: dict[BytesPair, set[int]],
        bytes_pair_counts: dict[BytesPair, int],
        search_heap: list[tuple[int, BytesPair]],
        pretoken_info_list: list[PretokenInfo],
    ) -> None:
        # TODO(djwenren): this could be further optimized by recording the nodes in the linked
        # list that contain the most frequent pair, so we don't have to traverse the entire list.
        new_token = most_frequent_pair.first + most_frequent_pair.second
        newly_generated_pair_counts: dict[BytesPair, int] = defaultdict(int)
        for pos in bytes_pair_to_pretoken_positions.get(most_frequent_pair, set()):
            pretoken_info = pretoken_info_list[pos]
            node = pretoken_info.first_node
            while node is not None:
                if node.bytes_pair != most_frequent_pair:
                    node = node.next
                    continue
                # Update the previous node's next pointer and bytes pair.
                if node.prev is None:
                    pretoken_info.first_node = node.next
                else:
                    node.prev.next = node.next
                    new_bytes_pair = BytesPair(
                        first=node.prev.bytes_pair.first,
                        second=new_token,
                    )
                    old_bytes_pair = node.prev.bytes_pair
                    node.prev.bytes_pair = new_bytes_pair
                    bytes_pair_counts[old_bytes_pair] -= pretoken_info.count
                    newly_generated_pair_counts[new_bytes_pair] += pretoken_info.count
                    bytes_pair_to_pretoken_positions[new_bytes_pair].add(pos)
                # Update the next node's previous pointer and bytes pair.
                if node.next is not None:
                    node.next.prev = node.prev
                    new_bytes_pair = BytesPair(
                        first=new_token,
                        second=node.next.bytes_pair.second,
                    )
                    old_bytes_pair = node.next.bytes_pair
                    node.next.bytes_pair = new_bytes_pair
                    bytes_pair_counts[old_bytes_pair] -= pretoken_info.count
                    newly_generated_pair_counts[new_bytes_pair] += pretoken_info.count
                    bytes_pair_to_pretoken_positions[new_bytes_pair].add(pos)
                # Move to the next node.
                node = node.next
        # Update counts for newly generated byte pairs.
        for bytes_pair, count in newly_generated_pair_counts.items():
            if count > 0:
                bytes_pair_counts[bytes_pair] += count
                heapq.heappush(
                    search_heap,
                    (
                        -count,
                        bytes_pair,
                    ),
                )
        # Remove the merged pair from counts and positions.
        del bytes_pair_counts[most_frequent_pair]
        del bytes_pair_to_pretoken_positions[most_frequent_pair]

    def get_merge_as_list_of_tuples(self) -> list[tuple[bytes, bytes]]:
        """Get the list of merges as a list of tuples."""
        return [(pair.first, pair.second) for pair in self._merges]

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        spelcial_tokens: list[str] | None = None,
    ) -> "BpeTokenizer":
        """Load a BPE tokenizer from saved vocab and merges files.

        Args:
            vocab_path (str): Path to the pickled vocab file.
            merges_path (str): Path to the pickled merges file.
            spelcial_tokens (list[str] | None): List of special tokens to include in the tokenizer.
        Returns:
            BpeTokenizer: The loaded BPE tokenizer.
        """
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)
        return cls(vocab=vocab, merges=merges, special_tokens=spelcial_tokens)

    def _init_for_encoding(self) -> None:
        """Initialize any data structures needed for encoding."""
        self._merges_to_idx = {merge: idx for idx, merge in enumerate(self._merges)}
        self._inverted_vocab = {
            token: token_id for token_id, token in self._vocab.items()
        }
        self._longest_vocab_length = max(len(token) for token in self._vocab.values())

    def encode(self, text: str) -> list[int]:
        """Encode the input text into a list of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: List of token IDs representing the encoded text.
        """
        if (not self._merges_to_idx) or (not self._inverted_vocab):
            self._init_for_encoding()
        output = []
        # Split by special tokens wrapped in capturing groups to keep the delimiters.
        for split_part in re.split("(" + self._special_tokens_split_regex + ")", text):
            if not split_part:
                continue
            if split_part in self._special_tokens:
                token_id = self._inverted_vocab[split_part.encode("utf-8")]
                output.append(token_id)
                continue
            for match in self._pretokenization_regex_pattern.finditer(split_part):
                pretoken = match.group(0).encode("utf-8")
                token_ids = self._encode_one_pretoken(pretoken)
                output.extend(token_ids)
        return output

    def encode_iterable(
        self, iterable: Iterable[str], chunk_size: int = 4096
    ) -> Iterator[int]:
        """Encode an iterable of texts into an iterator of token IDs."""
        if chunk_size < self._longest_vocab_length:
            raise ValueError(
                f"chunk_size must be at least {self._longest_vocab_length}"
            )
        current_chunk, lookahead = "", ""
        # i = 0
        if (not self._merges_to_idx) or (not self._inverted_vocab):
            self._init_for_encoding()
        for text in iterable:
            if len(current_chunk) < chunk_size:
                current_chunk += text
                continue
            if len(lookahead) < self._longest_vocab_length:
                lookahead += text
                continue
            tokens, num_chars_tokenized = self._encode_one_chunk(
                current_chunk, lookahead
            )
            yield from tokens
            current_chunk = (current_chunk + lookahead)[num_chars_tokenized:] + text
            lookahead = ""
        # Encode any remaining text.
        if current_chunk or lookahead:
            tokens = self.encode(current_chunk + lookahead)
            yield from tokens

    def _encode_one_chunk(
        self,
        chunk: str,
        lookahead: str,
    ) -> tuple[list[int], int]:
        """Encode a single chunk of text into a list of token IDs.

        Args:
            chunk (str): The input text chunk to encode.
            lookahead (str): Additional text to look ahead for tokenization. This is used to handle
                cases where a token may span across chunk boundaries.

        Returns:
            list[int]: List of token IDs representing the encoded text chunk.
            int: Number of characters from the chunk that were tokenized.
        """
        output = []
        num_characters_tokenized = 0
        main_chunk_size = len(chunk)
        # print(f"main_chunk_size: {main_chunk_size}, lookahead size: {len(lookahead)}")
        for split_part in re.split(
            "(" + self._special_tokens_split_regex + ")", chunk + lookahead
        ):
            if not split_part:
                continue
            if split_part in self._special_tokens:
                token_id = self._inverted_vocab[split_part.encode("utf-8")]
                output.append(token_id)
                num_characters_tokenized += len(split_part)
                if num_characters_tokenized >= main_chunk_size:
                    return output, num_characters_tokenized
                else:
                    continue
            for match in self._pretokenization_regex_pattern.finditer(split_part):
                pretoken = match.group(0).encode("utf-8")
                token_ids = self._encode_one_pretoken(pretoken)
                output.extend(token_ids)
                num_characters_tokenized += len(match.group(0))
                if num_characters_tokenized >= main_chunk_size:
                    return output, num_characters_tokenized
        return output, num_characters_tokenized

    def _encode_one_pretoken(self, pretoken: bytes) -> list[int]:
        """Encode a single pretoken into a list of token IDs.

        Args:
            pretoken (bytes): The pretoken to encode.

        Returns:
            list[int]: List of token IDs representing the encoded pretoken.
        """
        head_byte_pair = None
        min_merge_idx_heap: list[tuple[int, BytesPair]] = []
        byte_pairs_to_nodes: dict[BytesPair, list[BytesPairListNode]] = defaultdict(
            list
        )
        # Initialization.
        # Head's bytes pair always has b"" as first byte, so the head's bytes pair will never be
        # merged. This simplifies the logic when collecting final tokens.
        (
            head_byte_pair,
            min_merge_idx_heap,
            byte_pairs_to_nodes,
        ) = self._init_pretoken_for_encoding(pretoken)
        # BPE merging.
        while min_merge_idx_heap:
            _, bytes_pair_to_merge = heapq.heappop(min_merge_idx_heap)
            if bytes_pair_to_merge not in byte_pairs_to_nodes:
                continue
            for node in byte_pairs_to_nodes[bytes_pair_to_merge]:
                # Check if the node is still valid. If it has been merged, skip it.
                # The bytes pair in the min heap could be stale because we don't remove
                # entries from the heap when a bytes pair is merged.
                if node.bytes_pair != bytes_pair_to_merge:
                    continue
                new_token = bytes_pair_to_merge.first + bytes_pair_to_merge.second
                # Update the previous node's next pointer and bytes pair.
                if node.prev is None:
                    # This shouldn't happen since head's bytes pair always has b"" as first byte,
                    # and will never be popped from the heap.
                    head_byte_pair = node.next
                else:
                    node.prev.next = node.next
                    new_bytes_pair = BytesPair(
                        first=node.prev.bytes_pair.first,
                        second=new_token,
                    )
                    old_bytes_pair = node.prev.bytes_pair
                    node.prev.bytes_pair = new_bytes_pair
                    # Update the byte_pairs_to_nodes mapping.
                    byte_pairs_to_nodes[old_bytes_pair].remove(node.prev)
                    if not byte_pairs_to_nodes[old_bytes_pair]:
                        del byte_pairs_to_nodes[old_bytes_pair]
                    byte_pairs_to_nodes[new_bytes_pair].append(node.prev)
                    # If the new bytes pair is in merges, add it to the heap.
                    if new_bytes_pair in self._merges_to_idx:
                        merge_idx = self._merges_to_idx[new_bytes_pair]
                        heapq.heappush(min_merge_idx_heap, (merge_idx, new_bytes_pair))
                # Update the next node's previous pointer and bytes pair.
                if node.next is not None:
                    node.next.prev = node.prev
                    new_bytes_pair = BytesPair(
                        first=new_token,
                        second=node.next.bytes_pair.second,
                    )
                    old_bytes_pair = node.next.bytes_pair
                    node.next.bytes_pair = new_bytes_pair
                    # Update the byte_pairs_to_nodes mapping.
                    byte_pairs_to_nodes[old_bytes_pair].remove(node.next)
                    if not byte_pairs_to_nodes[old_bytes_pair]:
                        del byte_pairs_to_nodes[old_bytes_pair]
                    byte_pairs_to_nodes[new_bytes_pair].append(node.next)
                    # If the new bytes pair is in merges, add it to the heap.
                    if new_bytes_pair in self._merges_to_idx:
                        merge_idx = self._merges_to_idx[new_bytes_pair]
                        heapq.heappush(min_merge_idx_heap, (merge_idx, new_bytes_pair))
        # Collect the final tokens.
        return self._bytes_pair_linked_list_to_token_ids(head_byte_pair)

    def _init_pretoken_for_encoding(
        self,
        pretoken: bytes,
    ) -> tuple[
        BytesPairListNode | None,
        list[tuple[int, BytesPair]],
        dict[BytesPair, list[BytesPairListNode]],
    ]:
        """Initialize any data structures needed for encoding."""
        head_byte_pair = None
        min_merge_idx_heap: list[tuple[int, BytesPair]] = []
        byte_pairs_to_nodes: dict[BytesPair, list[BytesPairListNode]] = defaultdict(
            list
        )
        byte_sequence = list(pretoken)
        prev_node: BytesPairListNode | None = None
        for i, _ in enumerate(byte_sequence):
            bytes_pair = BytesPair(
                first=bytes([byte_sequence[i - 1]]) if i > 0 else b"",
                second=bytes([byte_sequence[i]]),
            )
            if (
                bytes_pair in self._merges_to_idx
                and bytes_pair not in byte_pairs_to_nodes
            ):
                merge_idx = self._merges_to_idx[bytes_pair]
                min_merge_idx_heap.append((merge_idx, bytes_pair))

            current_node = BytesPairListNode(bytes_pair=bytes_pair)
            byte_pairs_to_nodes[bytes_pair].append(current_node)
            if head_byte_pair is None:
                head_byte_pair = current_node
            if prev_node is not None:
                prev_node.next = current_node
                current_node.prev = prev_node
            prev_node = current_node
        heapq.heapify(min_merge_idx_heap)
        return head_byte_pair, min_merge_idx_heap, byte_pairs_to_nodes

    def _bytes_pair_linked_list_to_token_ids(
        self,
        head_byte_pair: BytesPairListNode | None,
    ) -> list[int]:
        """Convert a linked list of byte pairs to a list of token ids."""
        tokens: list[int] = []
        # Traverse the linked list and collect tokens.
        current_node = head_byte_pair
        while current_node is not None:
            # Head's bytes pair always has b"" as first byte, so we skip it.
            tokens.append(self._inverted_vocab[current_node.bytes_pair.second])
            current_node = current_node.next
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string.

        Args:
            token_ids (list[int]): List of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        byte_chunks = [self._vocab[token_id] for token_id in token_ids]
        decoded_string = b"".join(byte_chunks).decode("utf-8", errors="replace")
        return decoded_string

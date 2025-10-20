"""Byte Pair Encoding (BPE) tokenizer implementation."""

import heapq

from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool

import regex as re

from .pretokenization_example import find_chunk_boundaries


_DEFAULT_PRETOKENIZATION_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
_SPLIT_SPECIAL_TOKEN = "<|endoftext|>"


@dataclass
class BytesPairListNode:
    """A node in the bytes pair linked list."""

    bytes_pair: tuple[bytes, bytes]
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
        vocab_size: int,
        pretokenization_regex: str = _DEFAULT_PRETOKENIZATION_REGEX,
        split_special_token: str = _SPLIT_SPECIAL_TOKEN,
        special_tokens: list[str] | None = None,
        pretokenization_num_processes: int = 4,
    ) -> None:
        self._vocab_size = vocab_size
        # Compile the pretokenization regex pattern.
        self._pretokenization_regex_pattern = re.compile(
            pretokenization_regex, re.UNICODE
        )
        # Handle special tokens.
        self._split_special_token = split_special_token
        self._special_tokens = set(special_tokens or [])
        if self._split_special_token not in self._special_tokens:
            self._special_tokens.add(self._split_special_token)
        self._special_tokens_split_regex = "|".join(
            re.escape(token) for token in self._special_tokens
        )
        # Number of processes to use for pretokenization.
        self._pretokenization_num_processes = pretokenization_num_processes
        # Trained vocab and merges will be stored here after training.
        self._vocab: dict[int, bytes] = {
            i: special_token.encode("utf-8")
            for i, special_token in enumerate(sorted(self._special_tokens))
        }
        for i in range(256):
            self._vocab[i + len(self._special_tokens)] = bytes([i])
        self._merges: list[tuple[bytes, bytes]] = []

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
        heap: list[tuple[int, tuple[int, ...], tuple[bytes, bytes]]] = [
            (-count, self._get_inverted_ints_from_bytes_pair(bytes_pair), bytes_pair)
            for bytes_pair, count in bytes_pair_counts.items()
        ]
        heapq.heapify(heap)
        # Perform BPE merges until reaching the desired vocab size.
        while len(self._vocab) < self._vocab_size and heap:
            neg_count, bytes_pair_inverted_ints, most_frequent_pair = heapq.heappop(
                heap
            )
            current_count = bytes_pair_counts.get(most_frequent_pair, 0)
            # If the count has changed, reinsert with the updated count.
            if -neg_count != current_count:
                heapq.heappush(
                    heap, (-current_count, bytes_pair_inverted_ints, most_frequent_pair)
                )
                continue
            # Perform the merge.
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
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
        return self._vocab, self._merges

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
        dict[tuple[bytes, bytes], set[int]],
        dict[tuple[bytes, bytes], int],
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
        bytes_pair_to_pretoken_positions: dict[tuple[bytes, bytes], set[int]] = (
            defaultdict(set)
        )
        bytes_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)

        for pretoken, count in pretoken_counts.items():
            # Build the linked list of byte pairs for this pretoken
            first_node: BytesPairListNode | None = None
            prev_node: BytesPairListNode | None = None
            byte_sequence = list(pretoken)
            for i in range(len(byte_sequence) - 1):
                bytes_pair = (bytes([byte_sequence[i]]), bytes([byte_sequence[i + 1]]))
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
        most_frequent_pair: tuple[bytes, bytes],
        bytes_pair_to_pretoken_positions: dict[tuple[bytes, bytes], set[int]],
        bytes_pair_counts: dict[tuple[bytes, bytes], int],
        search_heap: list[tuple[int, tuple[int, ...], tuple[bytes, bytes]]],
        pretoken_info_list: list[PretokenInfo],
    ) -> None:
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        newly_generated_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
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
                    new_bytes_pair = (node.prev.bytes_pair[0], new_token)
                    old_bytes_pair = node.prev.bytes_pair
                    node.prev.bytes_pair = new_bytes_pair
                    bytes_pair_counts[old_bytes_pair] -= pretoken_info.count
                    newly_generated_pair_counts[new_bytes_pair] += pretoken_info.count
                    bytes_pair_to_pretoken_positions[new_bytes_pair].add(pos)
                # Update the next node's previous pointer and bytes pair.
                if node.next is not None:
                    node.next.prev = node.prev
                    new_bytes_pair = (new_token, node.next.bytes_pair[1])
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
                        self._get_inverted_ints_from_bytes_pair(bytes_pair),
                        bytes_pair,
                    ),
                )
        # Remove the merged pair from counts and positions.
        del bytes_pair_counts[most_frequent_pair]
        del bytes_pair_to_pretoken_positions[most_frequent_pair]

    def _get_inverted_ints_from_bytes_pair(
        self,
        bytes_pair: tuple[bytes, bytes],
    ) -> tuple[int, ...]:
        """Gets the inverted integer representation of a bytes pair."""

        def _bytes_to_inverted_int(b: bytes) -> tuple[int, ...]:
            return tuple(255 - byte for byte in list(b))

        return _bytes_to_inverted_int(bytes_pair[0]) + _bytes_to_inverted_int(
            bytes_pair[1]
        )

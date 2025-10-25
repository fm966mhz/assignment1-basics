"""Tokenize an input text file using a BPE tokenizer and save the token IDs."""

from multiprocessing import Pool

import numpy as np

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries


FLAGS = flags.FLAGS
flags.DEFINE_string("input_file_path", None, "Path to the input text file to tokenize.")
flags.DEFINE_string(
    "output_token_ids_path", None, "Path to save the token IDs (numpy .npy file)."
)
flags.DEFINE_string(
    "vocab_path", None, "Path to the pickled vocabulary file for the tokenizer."
)
flags.DEFINE_string(
    "merges_path", None, "Path to the pickled merges file for the tokenizer."
)
flags.DEFINE_integer(
    "num_tokenizer_processes",
    8,
    "Number of processes to use for tokenization.",
)
flags.DEFINE_string(
    "split_special_token",
    "<|endoftext|>",
    "Special token used to split the input during pretokenization.",
)
flags.DEFINE_integer(
    "num_input_file_chunks",
    16,
    "Number of chunks to split the input file into for parallel tokenization.",
)


def tokenize_one_chunk(
    tokenizer: BpeTokenizer,
    input_file_path: str,
    start: int,
    end: int,
) -> list[int]:
    """Tokenize a chunk of the input file from start to end byte offsets."""
    with open(input_file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode("utf-8")
        token_ids = tokenizer.encode(chunk_text)
    return token_ids


def main(argv):
    """Main function to tokenize the input file and save token IDs."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if FLAGS.input_file_path is None:
        raise app.UsageError("Must specify --input_file_path")

    if FLAGS.output_token_ids_path is None:
        raise app.UsageError("Must specify --output_token_ids_path")

    if FLAGS.vocab_path is None:
        raise app.UsageError("Must specify --vocab_path")

    if FLAGS.merges_path is None:
        raise app.UsageError("Must specify --merges_path")

    with open(FLAGS.input_file_path, "rb") as f:
        all_chunk_boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=FLAGS.num_input_file_chunks,
            split_special_token=FLAGS.split_special_token.encode("utf-8"),
        )

    logging.info("Loading BPE tokenizer...")
    tokenizer = BpeTokenizer.from_files(
        vocab_filepath=FLAGS.vocab_path,
        merges_filepath=FLAGS.merges_path,
    )

    logging.info("Tokenizing input file in parallel...")
    output_token_ids = []
    with Pool(processes=FLAGS.num_tokenizer_processes) as pool:
        token_ids_futures = []
        for start, end in zip(all_chunk_boundaries[:-1], all_chunk_boundaries[1:]):
            token_ids_futures.append(
                pool.apply_async(
                    tokenize_one_chunk,
                    args=(tokenizer, FLAGS.input_file_path, start, end),
                )
            )
        for future in tqdm(token_ids_futures):
            output_token_ids.extend(future.get())

    logging.info("Saving token IDs to output file...")
    np.save(FLAGS.output_token_ids_path, np.array(output_token_ids, dtype=np.uint16))
    logging.info("Tokenization complete.")


if __name__ == "__main__":
    app.run(main)

"""
Counts how often each token ID appears in one threshold's training file. The
counts decide which tokens are rare for the rare-token held-out loss, and
comparing the two thresholds' counts shows whether 0.6 sees rare tokens more
often than 0.9. Both are worked out later, from the files saved here.

Workers:

  count   a pool of processes (--workers), each given an equal range of the
          file. A worker reads --block-gb at a time, counts it, adds that to
          its running total, and sends the total back when its range ends
  main    adds the workers' totals, checks they add up to the file's token
          count, and saves the result

Peak memory is about 5 x --block-gb per worker, since counting turns each
2-byte token ID into 8 bytes: the defaults peak around 40 GB. Give the job 64 GB.

Runs in the training environment (~/venvs/train on HiPerGator), on CPU, from the
repo folder. Run it once per threshold.

Usage:
  python -u src/training/data/count_tokens.py --threshold 0.6 [--workers 8] [--block-gb 1]

Reads (data/source/training/, from tokenize_pages.py):
  train_<threshold>.bin   token IDs, flat uint16

Outputs (data/source/training/):
  token_counts_<threshold>.npy   49,152 uint64 counts; counts[i] is how many
                                 times token ID i appears

Prints: total tokens counted, and how many token IDs never appear.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

VOCAB_SIZE = 49152
TOKEN_BYTES = 2
DATA = "data/source/training"


# counts token IDs start up to end (positions in the file, in tokens), reading
# block_tokens at a time; returns one count per token ID
def count_range(path, start, end, block_tokens):
    counts = np.zeros(VOCAB_SIZE, dtype=np.uint64)
    with open(path, "rb") as f:
        f.seek(start * TOKEN_BYTES)
        for pos in range(start, end, block_tokens):
            block = np.fromfile(f, dtype=np.uint16, count=min(block_tokens, end - pos))
            counts += np.bincount(block, minlength=VOCAB_SIZE).astype(np.uint64)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", choices=["0.6", "0.9"], required=True)
    ap.add_argument("--workers", type=int, default=8, help="ranges of the file counted at once")
    ap.add_argument("--block-gb", type=float, default=1, help="GB each worker reads at a time")
    args = ap.parse_args()

    path = f"{DATA}/train_{args.threshold}.bin"
    n_tokens = os.path.getsize(path) // TOKEN_BYTES
    block_tokens = int(args.block_gb * 1e9) // TOKEN_BYTES

    # edges[i] to edges[i + 1] is worker i's range
    edges = [n_tokens * i // args.workers for i in range(args.workers + 1)]

    counts = np.zeros(VOCAB_SIZE, dtype=np.uint64)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for worker_counts in pool.map(count_range, [path] * args.workers, edges[:-1], edges[1:],
                                      [block_tokens] * args.workers):
            counts += worker_counts

    if counts.sum() != n_tokens:
        sys.exit(f"counted {counts.sum():,} tokens, but {path} has {n_tokens:,}; nothing saved")

    out = f"{DATA}/token_counts_{args.threshold}.npy"
    np.save(out, counts)
    print(f"{args.threshold}: {n_tokens:,} tokens, {np.sum(counts == 0):,} token IDs never appear")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

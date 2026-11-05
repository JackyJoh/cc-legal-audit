"""
Sorts token IDs into rarity buckets for the rare-token held-out loss, and prints
the mechanism check: does the 0.6 run see rare tokens more often than the 0.9
run?

Token IDs are sorted by their count in --threshold's training file, then cut so
each bucket holds about 1/N_BUCKETS of the text. Bucket 0 is the rarest. The most
common bucket is less even, since a single token can be a large share of the text.
The end-of-page token and IDs never seen in that file get -1 and aren't in any
bucket.

The usual buckets come from --threshold 0.6, since both training sets stem from
that file. --threshold 0.9 gives the robustness check's buckets.

Check: the ratio is a bucket's count in train_0.6 divided by its expected count
in the 0.9 run, which is its count in train_0.9 x the share of train_0.9 the run
reads. Both runs read the same number of tokens, so the ratios average about 1.
Above 1 in the rare buckets and below 1 in the common ones means 0.6 sees more
rare tokens. The table doesn't change the buckets.

Runs in the training environment (~/venvs/train on HiPerGator), from the repo
folder.

Usage:
  python src/training/data/rarity_buckets.py --threshold 0.6

Reads (data/source/training/, from count_tokens.py):
  token_counts_0.6.npy, token_counts_0.9.npy

Outputs (data/source/training/):
  rarity_buckets_<threshold>.npy   49,152 int8s; buckets[i] is token ID i's
                                   bucket, 0 to N_BUCKETS - 1, or -1 if left out

Prints: the share of train_0.9 its run reads, then per bucket the token types,
the smallest count and the ratio.
"""
import argparse

import numpy as np
from transformers import AutoTokenizer

VOCAB_SIZE = 49152
N_BUCKETS = 10
TOKENIZER = "HuggingFaceTB/SmolLM2-360M"
DATA = "data/source/training"

ap = argparse.ArgumentParser()
ap.add_argument("--threshold", choices=["0.6", "0.9"], required=True,
                help="training file whose counts decide the buckets")
args = ap.parse_args()

counts_06 = np.load(f"{DATA}/token_counts_0.6.npy").astype(np.float64)
counts_09 = np.load(f"{DATA}/token_counts_0.9.npy").astype(np.float64)
counts = counts_06 if args.threshold == "0.6" else counts_09

keep = counts > 0
keep[AutoTokenizer.from_pretrained(TOKENIZER).eos_token_id] = False

# kept token IDs, rarest first
ids = np.where(keep)[0]
ids = ids[np.argsort(counts[ids], kind="stable")]

# each ID's bucket comes from the share of the text up to and including it
text_share = np.cumsum(counts[ids]) / counts[ids].sum()
buckets = np.full(VOCAB_SIZE, -1, dtype=np.int8)
buckets[ids] = np.minimum((text_share * N_BUCKETS).astype(int), N_BUCKETS - 1)

read_share = counts_06.sum() / counts_09.sum()
expected_09 = counts_09 * read_share
print(f"share of train_0.9 read: {read_share:.4f}")
print("bucket  types  smallest_count  ratio_0.6/0.9")
for b in range(N_BUCKETS):
    types = np.where(buckets == b)[0]
    if len(types):
        ratio = counts_06[types].sum() / expected_09[types].sum()
        print(f"{b:6}  {len(types):5}  {int(counts[types].min()):14,}  {ratio:.4f}")

out = f"{DATA}/rarity_buckets_{args.threshold}.npy"
np.save(out, buckets)
print(f"saved {out}")

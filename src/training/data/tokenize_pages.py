"""
Turns each training set's pages into token IDs with SmolLM2's tokenizer (already
built; nothing is learned), one pages/ file at a time, so training never has to
read text.

Which pages go where:

  train_0.6   pages kept at 0.6, minus the test pages (heldout.py)
  train_0.9   pages kept at 0.9, minus the test pages, minus drop_0.9 (near-
              copies of test pages that 0.9 kept)

Each page's IDs are followed by the end-of-text ID, in the order the pages
appear in the file; training reads 2,048-ID chunks in random order, so pages
need no shuffling. Each page is tokenized once and added to whichever sets it
belongs in.

Workers:

  tokenize   a pool of processes (--workers), each taking one pages/ file at a
             time and writing that file's two outputs

A file is skipped once its 0.9 output exists, so a rerun resumes; outputs are
written under a temp name and swapped in, so a crash never leaves a
half-written file.

Runs in the training environment (~/venvs/train on HiPerGator), from the repo
folder.

Usage:
  python src/training/data/tokenize_pages.py [--workers 32]

Outputs (data/source/training/tokens/), one pair per pages/ file:
  06/<file>.bin   train_0.6's token IDs, flat uint16
  09/<file>.bin   train_0.9's token IDs, flat uint16; written last, so it
                  marks the file done

Prints: files left to tokenize, then each file's token counts as it finishes.
"""
import argparse
import glob
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from transformers import AutoTokenizer

os.makedirs("data/source/training/tokens/06", exist_ok=True)
os.makedirs("data/source/training/tokens/09", exist_ok=True)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")

removed_06 = np.load("data/source/dedup/removed_0.6.npy")  # id, kept_id
removed_09 = np.load("data/source/dedup/removed_0.9.npy")  # id, kept_id
drop_09_pages = np.load("data/source/training/drop_09_pages.npy")
heldout_pages = np.load("data/source/training/heldout_pages.npy")

# pages that must not go in each training set, sorted for fast lookups
exclude = {
    0.6: np.sort(np.concatenate([removed_06["id"], heldout_pages])),
    0.9: np.sort(np.concatenate([removed_09["id"], heldout_pages, drop_09_pages])),
}


# True if the page goes in that threshold's training set
def is_in(threshold, page_id):
    ex = exclude[threshold]
    pos = np.searchsorted(ex, page_id)      # where page_id would sit in the sorted list
    return pos == len(ex) or ex[pos] != page_id


def tokenize_file(path):
    # training ids
    train_06_ids = []
    train_09_ids = []

    name = os.path.basename(path).removesuffix(".jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        pages = np.array([json.loads(line) for line in f if line.strip()])

        # tokenize each page for training sets
        for p in pages:
            tokens = tokenizer(p["text"])["input_ids"]
            if is_in(0.6, p["id"]):
                train_06_ids.extend(tokens)
                train_06_ids.append(tokenizer.eos_token_id)
            if is_in(0.9, p["id"]):
                train_09_ids.extend(tokens)
                train_09_ids.append(tokenizer.eos_token_id)

    # save tokenized data as bin
    train_06_tokens = np.array(train_06_ids, dtype=np.uint16)
    train_09_tokens = np.array(train_09_ids, dtype=np.uint16)

    # temp name, then swap in, so a crash never leaves a half-written file; the
    # 0.9 file goes last because it's what marks the file done
    for folder, tokens in (("06", train_06_tokens), ("09", train_09_tokens)):
        out = f"data/source/training/tokens/{folder}/{name}.bin"
        tokens.tofile(out + ".tmp")
        os.replace(out + ".tmp", out)
    return name, len(train_06_tokens), len(train_09_tokens)


ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=32, help="files tokenized at once")
args = ap.parse_args()

# every pages file, full path each, skipping files already done (their 0.9 .bin
# exists), so a rerun resumes; --workers files tokenized at once
def done(path):
    name = os.path.basename(path).removesuffix(".jsonl.gz")
    return os.path.exists(f"data/source/training/tokens/09/{name}.bin")


paths = [p for p in sorted(glob.glob("data/source/pages/*.jsonl.gz")) if not done(p)]
print(f"{len(paths):,} files left to tokenize")
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for name, n_06, n_09 in pool.map(tokenize_file, paths):
        print(f"{name}: {n_06:,} tokens (0.6), {n_09:,} tokens (0.9)")


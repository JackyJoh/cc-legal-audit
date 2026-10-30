"""
Turns each training set's pages into token IDs with SmolLM2's tokenizer (already
built; nothing is learned), one text file (data/source/pages/*.jsonl.gz) at a
time, so training never has to read text.

The text files still have duplicates: two pages with the same URL, or two pages
with the same text. pages.npy (from score_pairs.py) keeps one page from each
pair, and this script uses only the page pages.npy kept.

Which pages go where:

  train_0.6   pages kept at 0.6, minus the test pages (heldout.py)
  train_0.9   pages kept at 0.9, minus the test pages, minus drop_0.9 (near-
              copies of test pages that 0.9 kept)

Each page's token IDs are followed by the end-of-text ID, in the order the
pages appear in the text file; training reads 2,048-ID chunks in random order,
so pages need no shuffling. Each page is tokenized once and added to whichever
training sets it belongs in.

Workers:

  tokenize   a pool of processes (--workers), each taking one text file at a
             time and writing that file's two outputs

A text file is skipped once its 0.9 output exists, so a rerun resumes; outputs
are written under a temp name and renamed when complete, so a crash never
leaves a half-written file.

Runs in the training environment (~/venvs/train on HiPerGator), from the repo
folder.

Usage:
  python src/training/data/tokenize_pages.py [--workers 32]

Outputs (data/source/training/tokens/), one pair per text file:
  06/<file>.bin   train_0.6's token IDs, flat uint16
  09/<file>.bin   train_0.9's token IDs, flat uint16; written last, so it
                  marks the text file done

Prints: text files left to tokenize, then each file's token counts as it
finishes.
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

# page ids that must not go in each training set, sorted for fast lookups
exclude = {
    0.6: np.sort(np.concatenate([removed_06["id"], heldout_pages])),
    0.9: np.sort(np.concatenate([removed_09["id"], heldout_pages, drop_09_pages])),
}

# pages.npy has one row per page, no duplicates; its file column says which
# text file the page is in (a line number in files.txt). Sorted by id for fast
# lookups
pages_npy = np.load("data/source/dedup/pages.npy")
order = np.argsort(pages_npy["id"])
npy_ids = pages_npy["id"][order]
npy_files = pages_npy["file"][order]
with open("data/source/dedup/files.txt", encoding="utf-8") as f:
    file_index = {line.strip(): i for i, line in enumerate(f)}


# True if pages.npy has this page in this text file. Checking the file matters:
# two pages with the same URL have the same id, and pages.npy kept only one
def in_pages_npy(page_id, file_i):
    pos = np.searchsorted(npy_ids, page_id)
    return pos < len(npy_ids) and npy_ids[pos] == page_id and npy_files[pos] == file_i


# True if the page goes in that threshold's training set
def is_in(threshold, page_id):
    ex = exclude[threshold]
    pos = np.searchsorted(ex, page_id)      # where page_id would sit in the sorted list
    return pos == len(ex) or ex[pos] != page_id


def tokenize_file(path):
    # token ids for each training set
    train_06_ids = []
    train_09_ids = []

    name = os.path.basename(path).removesuffix(".jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        pages = np.array([json.loads(line) for line in f if line.strip()])

        # tokenize each page that pages.npy kept. If the same URL shows up twice
        # in this one text file, both pass in_pages_npy, so seen (ids already
        # used in this file) makes sure only the first one is used
        file_i = file_index[name]
        seen = set()
        for p in pages:
            if p["id"] in seen or not in_pages_npy(p["id"], file_i):
                continue
            seen.add(p["id"])
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

    # write under a temp name, then rename, so a crash never leaves a
    # half-written file; the 0.9 file goes last because it marks the file done
    for folder, tokens in (("06", train_06_tokens), ("09", train_09_tokens)):
        out = f"data/source/training/tokens/{folder}/{name}.bin"
        tokens.tofile(out + ".tmp")
        os.replace(out + ".tmp", out)
    return name, len(train_06_tokens), len(train_09_tokens)


ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=32, help="text files tokenized at once")
args = ap.parse_args()


# True if this text file's 0.9 .bin already exists, so a rerun skips it
def done(path):
    name = os.path.basename(path).removesuffix(".jsonl.gz")
    return os.path.exists(f"data/source/training/tokens/09/{name}.bin")


# every text file not done yet; --workers files tokenized at once
paths = [p for p in sorted(glob.glob("data/source/pages/*.jsonl.gz")) if not done(p)]
print(f"{len(paths):,} text files left to tokenize")
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for name, n_06, n_09 in pool.map(tokenize_file, paths):
        print(f"{name}: {n_06:,} tokens (0.6), {n_09:,} tokens (0.9)")

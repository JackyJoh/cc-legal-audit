"""
Turns each training set's pages into token IDs with SmolLM2's tokenizer (already
built; nothing is learned) and writes one file per training set, so training
never has to read text.

The text files (data/source/pages/*.jsonl.gz) still have duplicates: two pages
with the same URL, or two pages with the same text. pages.npy (from
score_pairs.py) keeps one page from each pair, and this script uses only the
page pages.npy kept.

Which pages go where:

  train_0.6   pages kept at 0.6, minus the test pages (heldout.py)
  train_0.9   pages kept at 0.9, minus the test pages, minus drop_0.9 (near-
              copies of test pages that 0.9 kept)

Each page's token IDs are followed by the end-of-text ID. Text files are added
in name order, and pages in the order they appear in their text file, so a
rerun gives the same bytes; training reads 2,048-ID chunks in random order, so
pages need no shuffling. Each page is tokenized once and added to whichever
training sets it belongs in.

Workers:

  tokenize   a pool of processes (--workers), each taking one text file at a
             time and sending its token IDs back
  main       appends each text file's token IDs to the two output files, in
             name order, then records it in progress.txt

A rerun resumes: each output file is cut back to the size progress.txt
records (removing anything half-appended when the last run stopped), and the
text files listed there are skipped.

Runs in the training environment (~/venvs/train on HiPerGator), from the repo
folder.

Usage:
  python src/training/data/tokenize_pages.py [--workers 32]

Outputs (data/source/training/):
  train_0.6.bin   train_0.6's token IDs, flat uint16
  train_0.9.bin   train_0.9's token IDs, flat uint16
  progress.txt    one line per text file added: name, tokens added to 0.6,
                  tokens added to 0.9

Prints: text files left to tokenize, each file's token counts as it's added,
then each training set's total tokens.
"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

import numpy as np
from transformers import AutoTokenizer

OUT = "data/source/training"
BIN_06 = f"{OUT}/train_0.6.bin"
BIN_09 = f"{OUT}/train_0.9.bin"
PROGRESS = f"{OUT}/progress.txt"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")

removed_06 = np.load("data/source/dedup/removed_0.6.npy")  # id, kept_id
removed_09 = np.load("data/source/dedup/removed_0.9.npy")  # id, kept_id
drop_09_pages = np.load(f"{OUT}/drop_09_pages.npy")
heldout_pages = np.load(f"{OUT}/heldout_pages.npy")

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


# one text file's token IDs for each training set; returns (name, 0.6 IDs, 0.9 IDs)
def tokenize_file(path):
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

    return name, np.array(train_06_ids, dtype=np.uint16), np.array(train_09_ids, dtype=np.uint16)


ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=32, help="text files tokenized at once")
args = ap.parse_args()

# text files already added, and how many tokens they put in each output file
done = set()
total_06 = total_09 = 0
if os.path.exists(PROGRESS):
    with open(PROGRESS, encoding="utf-8") as f:
        for line in f:
            name, n_06, n_09 = line.split()
            done.add(name)
            total_06 += int(n_06)
            total_09 += int(n_09)

# cut each output file back to what progress.txt records (2 bytes per token ID),
# removing anything half-appended when the last run stopped
for path, total in ((BIN_06, total_06), (BIN_09, total_09)):
    open(path, "ab").close()        # creates the file if it doesn't exist
    if os.path.getsize(path) < total * 2:
        sys.exit(f"{path} is smaller than progress.txt records; delete it and "
                 "progress.txt, then rerun")
    os.truncate(path, total * 2)

# every text file not added yet, in name order; --workers files tokenized at once
paths = sorted(glob.glob("data/source/pages/*.jsonl.gz"))
paths = [p for p in paths if os.path.basename(p).removesuffix(".jsonl.gz") not in done]
print(f"{len(paths):,} text files left to tokenize")

with open(BIN_06, "ab") as f_06, open(BIN_09, "ab") as f_09, \
        open(PROGRESS, "a", encoding="utf-8") as log, \
        ProcessPoolExecutor(max_workers=args.workers) as pool:
    # at most 2 x --workers text files in progress at once, so finished ones
    # can't pile up in memory if writing is slower than tokenizing; results are
    # taken in name order, whichever worker finishes first
    pending = deque()
    todo = iter(paths)
    for path in islice(todo, 2 * args.workers):
        pending.append(pool.submit(tokenize_file, path))
    while pending:
        name, ids_06, ids_09 = pending.popleft().result()
        for path in islice(todo, 1):
            pending.append(pool.submit(tokenize_file, path))

        # token IDs are written first, then progress.txt, so progress.txt never
        # records a text file whose IDs aren't fully written
        for f, ids in ((f_06, ids_06), (f_09, ids_09)):
            ids.tofile(f)
            f.flush()
        log.write(f"{name} {len(ids_06)} {len(ids_09)}\n")
        log.flush()

        total_06 += len(ids_06)
        total_09 += len(ids_09)
        print(f"{name}: {len(ids_06):,} tokens (0.6), {len(ids_09):,} tokens (0.9)")

print(f"Total: {total_06:,} tokens (0.6), {total_09:,} tokens (0.9)")

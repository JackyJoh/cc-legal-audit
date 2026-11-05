"""
Pulls the held-out pages' text out of the text files into one file, for
perplexity. heldout_pages.npy only has their IDs, and they're spread across
nearly all of data/source/pages/*.jsonl.gz.

Text files still have duplicates (see tokenize_pages.py), so a page is taken
only from the text file pages.npy says it's in, and only its first copy there.

Saves text, not token IDs, so perplexity can tokenize it for any model.

Workers:

  extract   a pool of processes (--workers). Each task is one text file and the
            held-out IDs in it; a worker reads the file until it has found them
            and sends those pages back, then takes the next task
  main      builds the tasks, checks every held-out page came back, and writes
            the output file, so a failed run leaves no output

Runs in the training environment (~/venvs/train on HiPerGator), on CPU, from the
repo folder.

Usage:
  python -u src/training/data/heldout_text.py [--workers 16]

Reads:
  data/source/training/heldout_pages.npy   held-out page IDs
  data/source/dedup/pages.npy              id, bucket, file, words
  data/source/dedup/files.txt              text file names; pages.npy's file is a
                                           line number here
  data/source/pages/*.jsonl.gz             the text files

Outputs (data/source/training/):
  heldout.jsonl.gz   one JSON line per held-out page: id, bucket (0 = general,
                     1 = legal), text; pages in text file order, so a rerun gives
                     the same file

Prints: text files to read, progress every PRINT_EVERY files, then pages written
per bucket.
"""
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DEDUP = "data/source/dedup"
PAGES = "data/source/pages"
OUT = "data/source/training"
PRINT_EVERY = 1000


# one task; returns (id, text) for each page in ids, stopping once all are found
def extract(path, ids):
    remaining = set(ids)
    found = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            page = json.loads(line)
            if page["id"] in remaining:
                remaining.remove(page["id"])
                found.append((page["id"], page["text"]))
                if not remaining:
                    break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16, help="text files read at once")
    args = ap.parse_args()

    heldout = np.load(f"{OUT}/heldout_pages.npy")
    pages = np.load(f"{DEDUP}/pages.npy")
    rows = pages[np.isin(pages["id"], heldout)]
    if len(rows) != len(heldout):
        sys.exit(f"pages.npy has {len(rows):,} of the {len(heldout):,} held-out pages")

    with open(f"{DEDUP}/files.txt", encoding="utf-8") as f:
        names = [line.strip() for line in f]

    # text file (line number in files.txt) -> held-out IDs in it
    ids_per_file = defaultdict(set)
    bucket_of = {}
    for page_id, file_i, bucket in zip(rows["id"].tolist(), rows["file"].tolist(),
                                       rows["bucket"].tolist()):
        ids_per_file[file_i].add(page_id)
        bucket_of[page_id] = bucket

    files = sorted(ids_per_file)
    paths = [f"{PAGES}/{names[i]}.jsonl.gz" for i in files]
    print(f"{len(files):,} text files to read")

    found = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = pool.map(extract, paths, [ids_per_file[i] for i in files])
        for done, pages_found in enumerate(tasks, 1):
            found.extend(pages_found)
            if done % PRINT_EVERY == 0:
                print(f"{done:,} / {len(files):,} text files, {len(found):,} pages")

    if len(found) != len(heldout):
        sys.exit(f"found {len(found):,} of the {len(heldout):,} held-out pages; nothing saved")

    out = f"{OUT}/heldout.jsonl.gz"
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for page_id, text in found:
            f.write(json.dumps({"id": page_id, "bucket": bucket_of[page_id], "text": text}) + "\n")

    per_bucket = Counter(bucket_of[page_id] for page_id, _ in found)
    print(f"{per_bucket[0]:,} general pages, {per_bucket[1]:,} legal pages")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

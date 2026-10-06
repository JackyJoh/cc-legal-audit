"""
Fingerprints the pages source_pages.py kept, one finished pages/ file at a time,
so the pairs step never has to read most pages' text again.

Per page (minhash.py):

  id      the page's ID, as source_pages.py wrote it
  text    one number for its words, ignoring case, spacing and punctuation;
          identical text, identical number (for exact copies)
  bands   its 128 MinHash numbers squashed into 32 band values (for near
          copies)

Workers:

  fingerprint   a pool of processes (--workers), each taking one finished
                pages/ file at a time, fingerprinting its pages and writing
                that file's two outputs; no other process ever writes them

Needs no labels, so it runs alongside label_pages.py.

A pages/ file is picked up once source_pages.py has finished it (its stats file
exists) and skipped once its fingerprints exist, so a rerun resumes; a file cut
off mid-way is fingerprinted again from the start. With --watch, the script
keeps checking for new files while source_pages.py runs, and stops once
source_pages.py has finished and every file is fingerprinted.

Usage:
  python src/corpus/dedup/fingerprint.py --out data/source [--watch]

Outputs (--out, source_pages.py's folder), one pair per pages/ file:
  fingerprints/<file>.bands.npy   the band values, 32 rows (one per band) x one
                                  column per page, so the pairs step can read a
                                  single band for every page
  fingerprints/<file>.npz         id and text hash per page, in the same page
                                  order; written last, so it marks the file done

Next:
  python src/corpus/dedup/score_pairs.py --out data/source
"""
import argparse
import glob
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from minhash import NUM_BANDS, band_keys, shingle_hashes, signature, text_hash

POLL    = 10                    # seconds between checks for new files
REFRESH = 0.5                   # seconds between progress bar updates


# pages/ files source_pages.py has finished: name -> how many pages it kept
def finished_files(out, known):
    for p in glob.glob(f"{out}/stats/*.json"):
        name = os.path.basename(p).removesuffix(".json")
        if name not in known:
            with open(p, encoding="utf-8") as f:
                known[name] = json.load(f).get("kept", 0)
    return known


# rows of one pages/ file
def read_pages(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# fingerprint one pages/ file and write its two outputs; returns (file, pages, seconds)
def fingerprint_file(out, name):
    start = time.time()
    ids, texts, bands = [], [], []
    for r in read_pages(f"{out}/pages/{name}.jsonl.gz"):
        ids.append(r["id"])
        texts.append(text_hash(r["text"]))
        bands.append(band_keys(signature(shingle_hashes(r["text"]))))
    base = f"{out}/fingerprints/{name}"
    # temp name, then swap in, so a crash never leaves a half-written file; the
    # .npz goes last because it's what marks the file done
    np.save(base + ".tmp.npy", np.array(bands, np.uint64).reshape(-1, NUM_BANDS).T.copy())
    os.replace(base + ".tmp.npy", base + ".bands.npy")
    np.savez(base + ".tmp.npz", id=np.array(ids, np.int64), text=np.array(texts, np.uint64))
    os.replace(base + ".tmp.npz", base + ".npz")
    return name, len(ids), time.time() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", default="data/source", help="source_pages.py's --out folder")
    ap.add_argument("--workers", type=int, default=4, help="processes, one pages/ file each")
    ap.add_argument("--watch", action="store_true", help="keep fingerprinting new files until source_pages.py is done")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/fingerprints", exist_ok=True)

    files, sent, running = {}, set(), set()
    n_files, n_done, skipped, pages, busy = 0, 0, 0, 0, 0.0
    last_scan = None
    columns = (TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
               TextColumn("{task.fields[info]}"), TimeRemainingColumn())
    with ProcessPoolExecutor(args.workers) as pool, Progress(*columns, speed_estimate_period=60) as bars:
        t_files = bars.add_task("files 0/0", total=None, info="")
        while True:
            # hand each finished pages/ file to a worker once per run, unless already fingerprinted
            if last_scan is None or time.monotonic() - last_scan >= POLL:
                last_scan = time.monotonic()
                for name in sorted(finished_files(args.out, files)):
                    if name not in sent:
                        sent.add(name)
                        if os.path.exists(f"{args.out}/fingerprints/{name}.npz"):
                            skipped += 1
                        else:
                            running.add(pool.submit(fingerprint_file, args.out, name))
                            n_files += 1
            # files whose fingerprints are written
            for job in [j for j in running if j.done()]:
                running.remove(job)
                name, n, secs = job.result()
                n_done, pages, busy = n_done + 1, pages + n, busy + secs
                bars.console.print(escape(f"  [{n_done}/{n_files}] {name[16:22]}-{name[-5:]}: "
                                          f"{n:,} pages in {secs:.0f}s"))
            # speed while working, per process, not counting time spent waiting for files
            rate = pages / busy if busy else 0
            bars.update(t_files, description=f"files {n_done}/{n_files}", completed=n_done,
                        total=n_files or None, info=f"{pages:,} pages  {rate:,.0f}/s each")

            # done when nothing is running and (with --watch) sourcing has finished too
            source_done = not args.watch or os.path.exists(f"{args.out}/source.done")
            if not running and source_done and set(finished_files(args.out, files)) <= sent:
                break
            time.sleep(REFRESH)

    if skipped:
        print(f"{skipped:,} files were already fingerprinted")
    print(f"\nthis run: {n_done:,} files, {pages:,} pages fingerprinted")


if __name__ == "__main__":
    main()

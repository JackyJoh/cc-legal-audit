"""
Finds which pages are near-copies of each other and how similar each pair is:
every pair scoring 0.6 or higher on word 13-gram Jaccard, after removing exact
copies. Runs once, after sourcing, labeling and fingerprinting have finished.
Removes nothing itself; remove_duplicates.py applies each threshold.

Four steps:

  1. load      every page's ID, text hash and word count (fingerprints/) and
               bucket (labels/); stops if a page has no label, a file's two
               fingerprint files disagree on its page count, or a file was
               fingerprinted before word counts were added
  2. copies    pages with the same URL, or the same text hash: one is kept
               (first seen for the same URL; lowest survivor_rank for the same
               text), the rest are dropped and logged
  3. bands     for each band position, pages with the same band value become
               candidate pairs; several bands run at once. Stops if one group
               sharing a value is bigger than --max-group, since checking all
               its pairs could take days
  4. scores    for every candidate pair, the exact Jaccard of the two pages'
               13-word phrases. Phrases aren't stored, so the text of pages in
               a candidate pair (and only those) is re-read from pages/ to
               rebuild them. Pairs at 0.6 or higher are kept

Workers:

  bands    a pool of processes (--band-workers), each taking one band position
           at a time (about 1 GB each per 60M pages)
  phrases  a pool of processes (--workers), each taking one pages/ file at a
           time and rebuilding the phrases of its candidate pages
  scores   a pool of processes (--workers), each scoring one block of
           candidate pairs at a time

Steps 3 and 4 save their work as they go, so a rerun picks up where the last
one stopped; delete dedup/work/ if the input changes.

Usage:
  python src/corpus/dedup/score_pairs.py --out data/source

Outputs (--out/dedup/):
  pairs.npy          every pair scoring >= 0.6: id_a, id_b, score
  pages.npy          every page left after exact copies: id, bucket
                     (0 = general, 1 = legal), file (its line in files.txt),
                     words
  files.txt          the pages/ file names, one per line, in file-number order
  exact_copies.npy   every dropped exact copy: id, kept_id, reason
                     (1 = same URL, 2 = same text)
  work/              each step's saved progress

Next:
  python src/corpus/dedup/remove_duplicates.py --out data/source
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from minhash import NUM_BANDS, exact_jaccard, shingle_hashes, survivor_rank

MIN_SCORE = 0.6
BUCKETS   = ("general", "legal")    # label bucket names -> codes 0, 1
SAME_URL, SAME_TEXT = 1, 2          # exact_copies.npy reason codes
MAX_GROUP = 20_000                  # ~200M pairs to check in one group
BLOCK     = 1_000_000               # candidate pairs per scoring task

PAIR = np.dtype([("id_a", "<i8"), ("id_b", "<i8"), ("score", "<f4")])
PAGE = np.dtype([("id", "<i8"), ("bucket", "u1"), ("file", "<u4"), ("words", "<u4")])
COPY = np.dtype([("id", "<i8"), ("kept_id", "<i8"), ("reason", "u1")])

console = Console()


# write to a temp name, then swap it in, so a crash never leaves a half-written file
def save(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue                         # a line cut off by a crash


def progress():
    return Progress(TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
                    TextColumn("{task.fields[info]}"), TimeRemainingColumn(),
                    console=console, speed_estimate_period=60)


# run fn over each item's args in a pool, advancing one bar; returns results in item order
def run_pool(desc, fn, items, workers, initializer=None, initargs=()):
    results = [None] * len(items)
    with progress() as bars, ProcessPoolExecutor(workers, initializer=initializer, initargs=initargs) as ex:
        task = bars.add_task(desc, total=len(items), info="")
        jobs = {ex.submit(fn, *args): i for i, args in enumerate(items)}
        for job in as_completed(jobs):
            results[jobs[job]] = job.result()
            bars.advance(task)
    return results


# step 1: every page's id, text hash, bucket, file number and word count, in
# fingerprint file order; returns them plus each file's first row position
def step1(out, names):
    parts = [np.load(f"{out}/fingerprints/{n}.npz") for n in names]
    for n, p in zip(names, parts):
        if "words" not in p:
            sys.exit(f"{n}: fingerprinted before word counts were added; "
                     "delete fingerprints/ and rerun fingerprint.py")
        cols = np.load(f"{out}/fingerprints/{n}.bands.npy", mmap_mode="r").shape[1]
        if cols != len(p["id"]):
            sys.exit(f"{n}: {len(p['id'])} pages in its .npz but {cols} in its .bands.npy; "
                     "delete both and rerun fingerprint.py")
    ids   = np.concatenate([p["id"] for p in parts])
    text  = np.concatenate([p["text"] for p in parts])
    words = np.concatenate([p["words"] for p in parts])
    file  = np.repeat(np.arange(len(parts), dtype=np.uint32), [len(p["id"]) for p in parts])
    offsets = np.cumsum([0] + [len(p["id"]) for p in parts[:-1]])

    # buckets from labels/, looked up by id
    lab_ids, lab_buckets = [], []
    for path in glob.glob(f"{out}/labels/*.jsonl"):
        for r in read_jsonl(path):
            lab_ids.append(r["id"])
            lab_buckets.append(BUCKETS.index(r["bucket"]))
    lab_ids, first = np.unique(np.array(lab_ids, np.int64), return_index=True)
    lab_buckets = np.array(lab_buckets, np.uint8)[first]
    k = np.minimum(np.searchsorted(lab_ids, ids), max(len(lab_ids) - 1, 0))
    labeled = (lab_ids[k] == ids) if len(lab_ids) else np.zeros(len(ids), bool)
    if not labeled.all():
        sys.exit(f"{int((~labeled).sum()):,} of {len(ids):,} pages have no label; "
                 "finish label_pages.py first")
    console.print(f"  {len(ids):,} pages in {len(names):,} files, all labeled")
    return ids, text, lab_buckets[k], file, words, offsets


# step 2: which pages survive exact copies; saves keep.npy, ids.npy and the outputs
def step2(ids, text, bucket, file, words, work, dedup):
    # same URL means same ID; the first one seen is kept
    keep = np.zeros(len(ids), bool)
    _, first = np.unique(ids, return_index=True)
    keep[first] = True
    url_drop = np.flatnonzero(~keep)

    # same text: sort by text, then rank; the first of each run of equal text is kept
    order = first[np.lexsort((survivor_rank(ids[first]), text[first]))]
    t = text[order]
    head = np.r_[True, t[1:] != t[:-1]]
    run_head = order[np.maximum.accumulate(np.where(head, np.arange(len(order)), 0))]
    text_drop, text_kept = order[~head], run_head[~head]
    keep[text_drop] = False

    copies = np.empty(len(url_drop) + len(text_drop), COPY)
    copies["id"]      = np.r_[ids[url_drop], ids[text_drop]]
    copies["kept_id"] = np.r_[ids[url_drop], ids[text_kept]]
    copies["reason"]  = np.r_[np.full(len(url_drop), SAME_URL), np.full(len(text_drop), SAME_TEXT)]
    pages = np.empty(int(keep.sum()), PAGE)
    pages["id"], pages["bucket"] = ids[keep], bucket[keep]
    pages["file"], pages["words"] = file[keep], words[keep]

    save(f"{work}/ids.npy", ids)
    save(f"{work}/keep.npy", keep)
    save(f"{dedup}/exact_copies.npy", copies)
    save(f"{dedup}/pages.npy", pages)
    console.print(f"  {len(url_drop):,} same-URL and {len(text_drop):,} same-text copies dropped, "
                  f"{len(pages):,} pages left")


# every pair within one group of page IDs, smaller ID first
def all_pairs(group):
    i, j = np.triu_indices(len(group), 1)
    a, b = group[i], group[j]
    return np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)


# step 3, one band position: every pair of kept pages sharing its value; returns
# the band, its biggest group sizes, and whether the pairs were written
def band_pairs(b, out, names, work, max_group):
    keep = np.load(f"{work}/keep.npy")
    ids  = np.load(f"{work}/ids.npy")[keep]
    vals = np.concatenate([np.load(f"{out}/fingerprints/{n}.bands.npy", mmap_mode="r")[b]
                           for n in names])[keep]
    order = np.argsort(vals, kind="stable")       # equal values end up side by side
    vals, ids = vals[order], ids[order]
    starts = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1]])
    sizes  = np.diff(np.r_[starts, len(vals)])
    top = np.sort(sizes)[::-1][:5].tolist()
    if top and top[0] > max_group:
        return b, top, False

    path = f"{work}/bands/{b:02d}.npy"
    if not os.path.exists(path):
        twos = starts[sizes == 2]                 # most groups are pairs: no loop needed
        pairs = [np.stack([np.minimum(ids[twos], ids[twos + 1]),
                           np.maximum(ids[twos], ids[twos + 1])], axis=1)]
        pairs += [all_pairs(ids[s:s + k]) for s, k in zip(starts[sizes > 2], sizes[sizes > 2])]
        save(path, np.concatenate(pairs))
    return b, top, True


# all bands' pairs, each pair once
def merge_bands(work):
    c = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{work}/bands/*.npy"))])
    c = np.unique(np.ascontiguousarray(c).view([("a", "<i8"), ("b", "<i8")]).ravel())
    return np.stack([c["a"], c["b"]], axis=1)


# step 4a, one pages/ file: the phrase hashes of its kept pages that are in a
# candidate pair, as one flat array plus where each page's phrases start and end
def file_phrases(out, name, offset, work):
    base = f"{work}/phrases/{name}"
    if os.path.exists(base + ".npz"):
        return
    wanted = np.load(f"{work}/candidate_ids.npy")
    ids = np.load(f"{out}/fingerprints/{name}.npz")["id"]
    keep = np.load(f"{work}/keep.npy", mmap_mode="r")[offset:offset + len(ids)]
    take = np.asarray(keep) & np.isin(ids, wanted)
    flat, pid, start, end, n = [], [], [], [], 0
    for r, t in zip(read_jsonl(f"{out}/pages/{name}.jsonl.gz"), take):
        if t:
            sh = shingle_hashes(r["text"])
            flat.append(sh)
            pid.append(r["id"])
            start.append(n)
            n += len(sh)
            end.append(n)
    save(base + ".npy", np.concatenate(flat) if flat else np.empty(0, np.uint64))
    save_npz(base + ".npz", id=np.array(pid, np.int64), start=np.array(start, np.int64),
             end=np.array(end, np.int64))


# step 4b lookup, loaded once per worker: page ID -> its phrase hashes
_lookup = {}


def scores_init(work, names):
    parts = [np.load(f"{work}/phrases/{n}.npz") for n in names]
    ids  = np.concatenate([p["id"] for p in parts])
    file = np.concatenate([np.full(len(p["id"]), i) for i, p in enumerate(parts)]).astype(np.int32)
    order = np.argsort(ids)
    _lookup.update(ids=ids[order], file=file[order], work=work, names=names, flats={},
                   start=np.concatenate([p["start"] for p in parts])[order],
                   end=np.concatenate([p["end"] for p in parts])[order])


def phrases(pid):
    L = _lookup
    k = np.searchsorted(L["ids"], pid)
    f = int(L["file"][k])
    if f not in L["flats"]:
        L["flats"][f] = np.load(f"{L['work']}/phrases/{L['names'][f]}.npy", mmap_mode="r")
    return L["flats"][f][L["start"][k]:L["end"][k]]


# step 4b, one block of candidate pairs: exact Jaccard, keeping pairs >= MIN_SCORE
def score_block(i, pairs, work):
    found = []
    for a, b in pairs:
        s = exact_jaccard(phrases(a), phrases(b))
        if s >= MIN_SCORE:
            found.append((a, b, s))
    save(f"{work}/scores/{i:05d}.npy", np.array(found, PAIR))


# wall-clock time since a monotonic() reading, as H:MM:SS
def took(since):
    return str(timedelta(seconds=round(time.monotonic() - since)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", default="data/source", help="source_pages.py's --out folder")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processes for phrases and scores")
    ap.add_argument("--band-workers", type=int, default=8, help="band positions processed at once (~1 GB each per 60M pages)")
    ap.add_argument("--max-group", type=int, default=MAX_GROUP, help="stop if pages sharing one band value exceed this")
    args = ap.parse_args()
    start = time.monotonic()                     # wall clock for the run; each step is timed too

    out, dedup = args.out, f"{args.out}/dedup"
    work = f"{dedup}/work"
    for d in ("bands", "phrases", "scores"):
        os.makedirs(f"{work}/{d}", exist_ok=True)
    names = sorted(os.path.basename(p).removesuffix(".npz") for p in glob.glob(f"{out}/fingerprints/*.npz"))
    sourced = {os.path.basename(p).removesuffix(".json") for p in glob.glob(f"{out}/stats/*.json")}
    if not names:
        sys.exit(f"no fingerprints in {out}/fingerprints; run fingerprint.py first")
    if missing := sourced - set(names):
        sys.exit(f"{len(missing):,} sourced files have no fingerprints; finish fingerprint.py first")

    console.print("[bold]1. load[/bold]")
    step = time.monotonic()
    ids, text, bucket, file, words, offsets = step1(out, names)
    with open(f"{dedup}/files.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    console.print(f"  took {took(step)}")

    console.print("[bold]2. exact copies[/bold]")
    step = time.monotonic()
    step2(ids, text, bucket, file, words, work, dedup)
    console.print(f"  took {took(step)}")

    console.print("[bold]3. bands[/bold]")
    step = time.monotonic()
    bands = run_pool("bands", band_pairs, [(b, out, names, work, args.max_group) for b in range(NUM_BANDS)],
                     args.band_workers)
    biggest = sorted(((k, b) for b, top, _ in bands for k in top), reverse=True)[:5]
    console.print("  biggest groups sharing a band value: "
                  + ", ".join(f"{k:,} (band {b})" for k, b in biggest))
    if too_big := [b for b, _, written in bands if not written]:
        sys.exit(f"  groups above --max-group {args.max_group:,} in bands {too_big}; "
                 "decide how to handle them before scoring")
    cands = merge_bands(work)
    save(f"{work}/candidate_ids.npy", np.unique(cands))
    console.print(f"  {len(cands):,} candidate pairs")
    console.print(f"  took {took(step)}")

    console.print("[bold]4. exact scores[/bold]")
    step = time.monotonic()
    run_pool("phrases", file_phrases, [(out, n, o, work) for n, o in zip(names, offsets)], args.workers)
    todo = [i for i in range(0, len(cands), BLOCK) if not os.path.exists(f"{work}/scores/{i // BLOCK:05d}.npy")]
    run_pool("scores", score_block, [(i // BLOCK, cands[i:i + BLOCK], work) for i in todo], args.workers,
             initializer=scores_init, initargs=(work, names))
    n_blocks = -(-len(cands) // BLOCK)
    pairs = np.concatenate([np.load(f"{work}/scores/{i:05d}.npy") for i in range(n_blocks)] or [np.empty(0, PAIR)])
    save(f"{dedup}/pairs.npy", pairs)
    console.print(f"  took {took(step)}")

    console.print(f"\n{len(pairs):,} pairs scoring >= {MIN_SCORE} -> {dedup}/pairs.npy")
    for t in (0.6, 0.7, 0.8, 0.9):
        console.print(f"  >= {t}: {int((pairs['score'] >= t).sum()):,}")
    console.print(f"took {took(start)}")


if __name__ == "__main__":
    main()

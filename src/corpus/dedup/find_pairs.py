"""
Finds every pair of pages scoring 0.6 or higher on word 13-gram Jaccard, after
removing exact copies. The expensive half of deduplication; group_pairs.py then
applies each threshold.

Five steps, each saved to disk, so a rerun picks up where the last one stopped:

  1. numbers   each page's ID, text hash, bucket and 32 band values
               (minhash.py), one worker per input file
  2. copies    pages with the same URL, or the same words once case, spacing
               and punctuation are ignored: one is kept (first seen for the
               same URL; lowest survivor_rank for the same text), the rest
               are dropped and logged
  3. bands     for each band, every two pages with the same band value become
               a candidate pair; several bands run at once
  4. phrases   the 13-word phrase hashes of each page in a candidate pair,
               rebuilt from the text, one worker per input file
  5. scores    exact Jaccard for every candidate pair; pairs at 0.6 or
               higher are kept

Every pair inside a group of pages sharing a band value is checked, however big
the group. The run prints the biggest groups, and stops before step 4 if one is
bigger than --max-group, since checking it could take days.

Input: a folder of .jsonl.gz files, one page per line, each page having passed
the quality filters (src/corpus/filters/pipeline.py):
  {"id": minhash.page_id(url), "url", "text", "bucket": "general" | "legal", ...}

Usage:
  python src/corpus/dedup/find_pairs.py --input data/pages --out data/dedup

Outputs (--out):
  pairs.npy          every pair scoring >= 0.6: id_a, id_b, score
  pages.npy          every page left after exact copies: id, bucket
                     (0 = general, 1 = legal)
  exact_copies.npy   every dropped exact copy: id, kept_id, reason
                     (1 = same URL, 2 = same text)
  work/              each step's saved files; delete it if the input changes

Next:
  python src/corpus/dedup/group_pairs.py --out data/dedup
"""
import argparse
import glob
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np

from minhash import (NUM_BANDS, band_keys, exact_jaccard, shingle_hashes,
                     signature, survivor_rank, text_hash)

MIN_SCORE = 0.6
BUCKETS   = ("general", "legal")    # input bucket names -> codes 0, 1
SAME_URL, SAME_TEXT = 1, 2          # exact_copies.npy reason codes
MAX_GROUP = 20_000                  # ~200M pairs to check in one group
BLOCK     = 1_000_000               # candidate pairs per scoring task

PAIR = np.dtype([("id_a", "<i8"), ("id_b", "<i8"), ("score", "<f4")])
PAGE = np.dtype([("id", "<i8"), ("bucket", "u1")])
COPY = np.dtype([("id", "<i8"), ("kept_id", "<i8"), ("reason", "u1")])


def read_pages(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stem(path):
    return os.path.basename(path).removesuffix(".jsonl.gz")


# write to a temp name, then swap it in, so a crash never leaves a half-written file
def save(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# step 1, one input file: each page's ID, text hash and bucket, and its band
# values stored band by band (32 x pages) so step 3 can read one band at a time
def step1_file(path, work):
    base = f"{work}/numbers/{stem(path)}"
    if os.path.exists(base + ".npz"):
        return
    ids, texts, buckets, bands = [], [], [], []
    for r in read_pages(path):
        ids.append(r["id"])
        texts.append(text_hash(r["text"]))
        buckets.append(BUCKETS.index(r["bucket"]))
        bands.append(band_keys(signature(shingle_hashes(r["text"]))))
    save(base + ".bands.npy", np.array(bands, np.uint64).reshape(-1, NUM_BANDS).T.copy())
    save_npz(base + ".npz", id=np.array(ids, np.int64), text=np.array(texts, np.uint64),
             bucket=np.array(buckets, np.uint8))


# step 2: which pages survive exact copies; returns each input file's first row
# position in the combined page list
def step2(stems, work, out):
    parts = [np.load(f"{work}/numbers/{s}.npz") for s in stems]
    ids    = np.concatenate([p["id"] for p in parts])
    text   = np.concatenate([p["text"] for p in parts])
    bucket = np.concatenate([p["bucket"] for p in parts])
    offsets = np.cumsum([0] + [len(p["id"]) for p in parts[:-1]])

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

    save(f"{work}/ids.npy", ids)
    save(f"{work}/keep.npy", keep)
    save(f"{out}/exact_copies.npy", copies)
    save(f"{out}/pages.npy", pages)
    print(f"  {len(ids):,} pages: {len(url_drop):,} same-URL copies and "
          f"{len(text_drop):,} same-text copies dropped, {len(pages):,} left")
    return offsets


# every pair within one group of page IDs, smaller ID first
def all_pairs(group):
    i, j = np.triu_indices(len(group), 1)
    a, b = group[i], group[j]
    return np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)


# step 3, one band: every pair of kept pages sharing this band's value; returns
# the band, its biggest group sizes, and whether the pairs were written
def step3_band(b, stems, work, max_group):
    keep = np.load(f"{work}/keep.npy")
    ids  = np.load(f"{work}/ids.npy")[keep]
    vals = np.concatenate([np.load(f"{work}/numbers/{s}.bands.npy", mmap_mode="r")[b]
                           for s in stems])[keep]
    order = np.argsort(vals, kind="stable")       # equal values end up side by side
    vals, ids = vals[order], ids[order]
    starts = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1]])
    sizes  = np.diff(np.r_[starts, len(vals)])
    top = np.sort(sizes)[::-1][:5].tolist()
    if top and top[0] > max_group:
        return b, top, False

    out = f"{work}/bands/{b:02d}.npy"
    if not os.path.exists(out):
        twos = starts[sizes == 2]                 # most groups are pairs: no loop needed
        pairs = [np.stack([np.minimum(ids[twos], ids[twos + 1]),
                           np.maximum(ids[twos], ids[twos + 1])], axis=1)]
        pairs += [all_pairs(ids[s:s + k]) for s, k in zip(starts[sizes > 2], sizes[sizes > 2])]
        save(out, np.concatenate(pairs))
    return b, top, True


# all bands' pairs, each pair once
def merge_bands(work):
    c = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{work}/bands/*.npy"))])
    c = np.unique(np.ascontiguousarray(c).view([("a", "<i8"), ("b", "<i8")]).ravel())
    return np.stack([c["a"], c["b"]], axis=1)


# step 4, one input file: the phrase hashes of its kept pages that are in a
# candidate pair, as one flat array plus where each page's phrases start and end
def step4_file(path, offset, work):
    base = f"{work}/phrases/{stem(path)}"
    if os.path.exists(base + ".npz"):
        return
    wanted = np.load(f"{work}/candidate_ids.npy")
    ids = np.load(f"{work}/numbers/{stem(path)}.npz")["id"]
    keep = np.load(f"{work}/keep.npy", mmap_mode="r")[offset:offset + len(ids)]
    take = np.asarray(keep) & np.isin(ids, wanted)
    flat, pid, start, end, n = [], [], [], [], 0
    for r, t in zip(read_pages(path), take):
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


# step 5 lookup, loaded once per worker: page ID -> its phrase hashes
_lookup = {}


def step5_init(work, stems):
    parts = [np.load(f"{work}/phrases/{s}.npz") for s in stems]
    ids  = np.concatenate([p["id"] for p in parts])
    file = np.concatenate([np.full(len(p["id"]), i) for i, p in enumerate(parts)]).astype(np.int32)
    order = np.argsort(ids)
    _lookup.update(ids=ids[order], file=file[order], work=work, stems=stems, flats={},
                   start=np.concatenate([p["start"] for p in parts])[order],
                   end=np.concatenate([p["end"] for p in parts])[order])


def phrases(pid):
    L = _lookup
    k = np.searchsorted(L["ids"], pid)
    f = int(L["file"][k])
    if f not in L["flats"]:
        L["flats"][f] = np.load(f"{L['work']}/phrases/{L['stems'][f]}.npy", mmap_mode="r")
    return L["flats"][f][L["start"][k]:L["end"][k]]


# step 5, one block of candidate pairs: exact Jaccard, keeping pairs >= MIN_SCORE
def step5_block(i, pairs, work):
    found = []
    for a, b in pairs:
        s = exact_jaccard(phrases(a), phrases(b))
        if s >= MIN_SCORE:
            found.append((a, b, s))
    save(f"{work}/scores/{i:05d}.npy", np.array(found, PAIR))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--input", required=True, help="folder of filtered .jsonl.gz page files")
    ap.add_argument("--out", default="data/dedup")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processes for steps 1, 4, 5")
    ap.add_argument("--band-workers", type=int, default=8, help="bands processed at once (~1 GB each per 60M pages)")
    ap.add_argument("--max-group", type=int, default=MAX_GROUP, help="stop if pages sharing one band value exceed this")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.input}/*.jsonl.gz"))
    if not files:
        sys.exit(f"no .jsonl.gz files in {args.input}")
    stems = [stem(f) for f in files]
    work = f"{args.out}/work"
    for d in ("numbers", "bands", "phrases", "scores"):
        os.makedirs(f"{work}/{d}", exist_ok=True)

    print(f"1. numbers: {len(files):,} input files")
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(step1_file, files, repeat(work)))

    print("2. copies")
    offsets = step2(stems, work, args.out)

    print("3. bands")
    with ProcessPoolExecutor(args.band_workers) as ex:
        bands = list(ex.map(step3_band, range(NUM_BANDS), repeat(stems), repeat(work), repeat(args.max_group)))
    biggest = sorted(((k, b) for b, top, _ in bands for k in top), reverse=True)[:5]
    print("  biggest groups sharing a band value: "
          + ", ".join(f"{k:,} (band {b})" for k, b in biggest))
    too_big = [b for b, _, written in bands if not written]
    if too_big:
        sys.exit(f"  groups above --max-group {args.max_group:,} in bands {too_big}; "
                 "decide how to handle them before scoring")
    cands = merge_bands(work)
    save(f"{work}/candidate_ids.npy", np.unique(cands))
    print(f"  {len(cands):,} candidate pairs")

    print("4. phrases")
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(step4_file, files, offsets, repeat(work)))

    print("5. scores")
    todo = [i for i in range(0, len(cands), BLOCK) if not os.path.exists(f"{work}/scores/{i // BLOCK:05d}.npy")]
    with ProcessPoolExecutor(args.workers, initializer=step5_init, initargs=(work, stems)) as ex:
        list(ex.map(step5_block, [i // BLOCK for i in todo], [cands[i:i + BLOCK] for i in todo], repeat(work)))
    n_blocks = -(-len(cands) // BLOCK)
    pairs = np.concatenate([np.load(f"{work}/scores/{i:05d}.npy") for i in range(n_blocks)] or [np.empty(0, PAIR)])
    save(f"{args.out}/pairs.npy", pairs)

    print(f"\n{len(pairs):,} pairs scoring >= {MIN_SCORE} -> {args.out}/pairs.npy")
    for t in (0.6, 0.7, 0.8, 0.9):
        print(f"  >= {t}: {int((pairs['score'] >= t).sum()):,}")


if __name__ == "__main__":
    main()
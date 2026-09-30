"""
Finds pairs of pages that MinHash would call near-duplicates, within the legal
set and within the control set.

How similarity is measured. Each page becomes the set of every 13-word phrase
in it. Two pages' similarity (Jaccard) is the share of phrases they have in
common: 1.0 is identical, 0.8 is the usual dedup threshold. MinHash is a fast
estimate of that number, used to skip pairs that are obviously not close; the
pairs it lets through get their exact similarity, and those at 0.6 or above
are kept.

Gopher filters first. Before any comparison, pages go through Gopher's quality
and repetition filters (src/corpus/filters/gopher.py: too short, mostly
symbols, heavily repeated text, and so on), since a real pipeline only
deduplicates pages that pass them. The run prints how many pages each filter
drops, per set.

Usage:
  python src/corpus/minhash_pilot/minhash_pairs.py

Outputs (data/minhash_pilot/):
  pairs.jsonl   every pair at similarity >= 0.6: which set, the two URLs and
                sites, and the exact and estimated similarity; no text
"""
import hashlib
import os
import re
import sys
from collections import Counter

import numpy as np

from sampling import OUT_DIR, load_jsonl, site_of, write_jsonl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "filters"))
from gopher import gopher_reject  # noqa: E402

SETS = {"legal":   f"{OUT_DIR}/legal_set.jsonl",
        "control": f"{OUT_DIR}/control_text.jsonl"}
OUTPUT = f"{OUT_DIR}/pairs.jsonl"

NGRAM       = 13
NUM_PERM    = 128
PREFILTER   = 0.5     # MinHash estimate needed to compute exact Jaccard
MIN_JACCARD = 0.6     # lowest threshold in the sweep
BANDS       = (0.6, 0.7, 0.8, 0.9)

# permutation h(x) = (a*x + b) mod P over 32-bit shingle hashes; a, b < 2^31
# keeps a*x + b inside uint64
P = np.uint64(4_294_967_311)                     # prime just above 2^32
_rng = np.random.default_rng(42)
A = _rng.integers(1, 2**31, NUM_PERM, dtype=np.uint64)
B = _rng.integers(0, 2**31, NUM_PERM, dtype=np.uint64)


def shingle_hashes(text):
    """Sorted unique 64-bit hashes of the page's word 13-grams."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < NGRAM:
        grams = {" ".join(words)}                # short page: one shingle
    else:
        grams = {" ".join(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}
    return np.unique(np.array(
        [int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
         for g in grams], dtype=np.uint64))


# MinHash signature: the minimum of each of the NUM_PERM permutations over the shingles
def signature(hashes):
    x = hashes & np.uint64(0xFFFFFFFF)
    return ((A[:, None] * x[None, :] + B[:, None]) % P).min(axis=1)


# true Jaccard of two sorted unique shingle-hash arrays
def exact_jaccard(a, b):
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


# all pairs within one set at exact Jaccard >= MIN_JACCARD: compare MinHash
# signatures first, compute exact Jaccard only for pairs passing PREFILTER
def pairs_in_set(name, rows):
    shingles = [shingle_hashes(r["text"]) for r in rows]
    sigs = np.stack([signature(s) for s in shingles])
    found = []
    for i in range(len(rows) - 1):
        est = (sigs[i + 1:] == sigs[i]).mean(axis=1)
        for j in np.nonzero(est >= PREFILTER)[0] + i + 1:
            jac = exact_jaccard(shingles[i], shingles[j])
            if jac >= MIN_JACCARD:
                a, b = rows[i], rows[j]
                found.append({"set": name, "url_a": a["url"], "url_b": b["url"],
                              "site_a": site_of(a["url"]), "site_b": site_of(b["url"]),
                              "jaccard": round(jac, 4), "minhash": round(float(est[j - i - 1]), 4)})
        if (i + 1) % 500 == 0:
            print(f"  {name}: {i + 1:,}/{len(rows):,} pages")
    return found


# highest band threshold that Jaccard j reaches (e.g. 0.74 -> 0.7)
def band_of(j):
    return max(b for b in BANDS if j >= b)


# find pairs in each set, write them out, print pair counts per band and the
# share of pairs within a single site
def main():
    pairs = []
    for name, path in SETS.items():
        rows = [r for r in load_jsonl(path) if r.get("text")]
        why = [gopher_reject(r["text"]) for r in rows]
        reasons = Counter(why)
        kept = [r for r, w in zip(rows, why) if w is None]
        print(f"{name}: {len(rows):,} pages, {len(kept):,} pass Gopher")
        for reason, c in reasons.most_common():
            if reason:
                print(f"  dropped {c:>5,}  {reason}")
        pairs += pairs_in_set(name, kept)
    write_jsonl(OUTPUT, pairs)
    print(f"\n{len(pairs):,} pairs at Jaccard >= {MIN_JACCARD} -> {OUTPUT}")

    print(f"\n  {'set':<9}" + "".join(f"{f'{b:.1f}+':>9}" for b in BANDS) + f"{'same site':>11}")
    for name in SETS:
        mine = [p for p in pairs if p["set"] == name]
        counts = [sum(1 for p in mine if band_of(p["jaccard"]) == b) for b in BANDS]
        same = sum(1 for p in mine if p["site_a"] == p["site_b"])
        share = f"{same / len(mine):.0%}" if mine else "-"
        print(f"  {name:<9}" + "".join(f"{c:>9,}" for c in counts) + f"{share:>11}")


if __name__ == "__main__":
    main()

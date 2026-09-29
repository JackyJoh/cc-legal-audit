"""
Finds near-duplicate page pairs within each pilot set (legal, control).

Pages are lowercased word 13-gram shingle sets. MinHash (128 permutations)
compares every pair in a set; pairs it estimates at >= 0.5 get their exact
Jaccard computed, and pairs at exact >= 0.6 are written. Bands downstream use
the exact value.

Reads data/pilot/legal_set.jsonl and data/pilot/control_text.jsonl.
Writes data/pilot/pairs.jsonl: {set, url_a, url_b, site_a, site_b, jaccard, minhash}.
"""
import hashlib
import re

import numpy as np

from sampling import OUT_DIR, load_jsonl, site_of, write_jsonl

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


def signature(hashes):
    x = hashes & np.uint64(0xFFFFFFFF)
    return ((A[:, None] * x[None, :] + B[:, None]) % P).min(axis=1)


def exact_jaccard(a, b):
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


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


def band_of(j):
    return max(b for b in BANDS if j >= b)


def main():
    pairs = []
    for name, path in SETS.items():
        rows = [r for r in load_jsonl(path) if r.get("text")]
        print(f"{name}: {len(rows):,} pages")
        pairs += pairs_in_set(name, rows)
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

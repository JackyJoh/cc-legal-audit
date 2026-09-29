"""
Finds near-duplicate page pairs within each pilot set (legal, control).

Pages are lowercased word 13-gram shingle sets. MinHash (128 permutations)
compares every pair in a set; pairs it estimates at >= 0.5 get their exact
Jaccard computed, and pairs at exact >= 0.6 are written. Bands downstream use
the exact value.

Before pairing, both sets go through Gopher's quality and repetition filters
(Rae et al. 2021, MassiveText; thresholds as in datatrove), because the full
pipeline dedups only pages that pass them. Paragraph-repetition rules are
skipped: the extracted text has line breaks but no blank lines.

Reads data/pilot/legal_set.jsonl and data/pilot/control_text.jsonl.
Writes data/pilot/pairs.jsonl: {set, url_a, url_b, site_a, site_b, jaccard, minhash}.
"""
import hashlib
import re
from collections import Counter

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

# Gopher quality filter
MIN_WORDS, MAX_WORDS   = 50, 100_000
MIN_MEAN_WORD, MAX_MEAN_WORD = 3, 10
MAX_SYMBOL_RATIO       = 0.1     # ('#' + ellipses) per word
MAX_BULLET_LINES       = 0.9
MAX_ELLIPSIS_LINES     = 0.3
MIN_ALPHA_WORDS        = 0.8     # words with at least one letter
MIN_STOP_WORDS         = 2
STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}

# Gopher repetition filter
MAX_DUP_LINES      = 0.30        # share of lines that repeat an earlier line
MAX_DUP_LINE_CHARS = 0.20        # share of characters in those lines
TOP_NGRAM_MAX = {2: 0.20, 3: 0.18, 4: 0.16}
DUP_NGRAM_MAX = {5: 0.15, 6: 0.14, 7: 0.13, 8: 0.12, 9: 0.11, 10: 0.10}
BULLETS = ("•", "●", "○", "▪", "-", "*", "–", "‣")
ELLIPSES = ("...", "…")


# characters in the most frequent n-gram, times how often it occurs
def top_ngram_chars(words, n):
    grams = Counter(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    if not grams:
        return 0
    gram, count = grams.most_common(1)[0]
    return len(gram) * count


# characters in n-grams that repeat an earlier n-gram, stepping past each repeat
# so overlapping repeats are counted once
def dup_ngram_chars(words, n):
    seen, chars, i = set(), 0, 0
    while i <= len(words) - n:
        gram = " ".join(words[i:i + n])
        if gram in seen:
            chars += len(gram)
            i += n
        else:
            seen.add(gram)
            i += 1
    return chars


# first Gopher rule the page fails, or None if it passes all of them
def gopher_reject(text):
    words = text.split()
    n = len(words)
    if not MIN_WORDS <= n <= MAX_WORDS:
        return "word count"
    if not MIN_MEAN_WORD <= sum(map(len, words)) / n <= MAX_MEAN_WORD:
        return "mean word length"
    if (text.count("#") + sum(text.count(e) for e in ELLIPSES)) / n > MAX_SYMBOL_RATIO:
        return "symbol ratio"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if sum(l.startswith(BULLETS) for l in lines) / len(lines) > MAX_BULLET_LINES:
        return "bullet lines"
    if sum(l.endswith(ELLIPSES) for l in lines) / len(lines) > MAX_ELLIPSIS_LINES:
        return "ellipsis lines"
    if sum(bool(re.search(r"[^\W\d_]", w)) for w in words) / n < MIN_ALPHA_WORDS:
        return "alphabetic words"
    if len(STOP_WORDS & {w.lower() for w in words}) < MIN_STOP_WORDS:
        return "stop words"

    seen, dup_lines, dup_chars = set(), 0, 0
    for l in lines:
        if l in seen:
            dup_lines += 1
            dup_chars += len(l)
        seen.add(l)
    if dup_lines / len(lines) > MAX_DUP_LINES:
        return "duplicate lines"
    total = len(text)
    if dup_chars / total > MAX_DUP_LINE_CHARS:
        return "duplicate line chars"
    for k, cap in TOP_NGRAM_MAX.items():
        if top_ngram_chars(words, k) / total > cap:
            return f"top {k}-gram"
    for k, cap in DUP_NGRAM_MAX.items():
        if dup_ngram_chars(words, k) / total > cap:
            return f"duplicate {k}-grams"
    return None


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

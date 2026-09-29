"""
Builds the hand-labeling batch for the MinHash pilot.

Draws up to --per-band pairs per (set, Jaccard band) from pairs.jsonl, bands
--min-band (0.7) and up, at most --site-cap (10) per site pair and one per
page within a band, shuffles legal and control together, and shows each pair
as a word diff: only what differs, with a few words of context. Pairs whose
text is identical are auto-labeled same (auto: true).

Writes to data/pilot/:
  pair_batch.jsonl   {pair, url_a, url_b, start, words_a, words_b, diff, label, auto}
  pair_scores.jsonl  {pair, set, band, jaccard, minhash, site_a, site_b}
Labels: same / version / different / exclude. Scores stay out of the batch.
When rebuilding over a labeled batch, hand labels carry over by URL pair;
labels on pairs that leave the batch are dropped.
"""
import argparse
import difflib
import hashlib
import os
import random
from collections import Counter, defaultdict

from minhash_pairs import BANDS, OUTPUT as PAIRS, SETS, band_of
from sampling import OUT_DIR, SEED, load_jsonl, write_jsonl

BATCH  = f"{OUT_DIR}/pair_batch.jsonl"
SCORES = f"{OUT_DIR}/pair_scores.jsonl"

CONTEXT    = 8        # words shown either side of a change
MAX_SIDE   = 300      # chars shown per side of one change
MAX_HUNKS  = 15       # changes shown per pair
MAX_WORDS  = 20_000   # words per page fed to the diff
START_WORDS = 40      # opening words of page A, to show what the document is


def seeded_order(key):
    return hashlib.sha1((key + SEED).encode("utf-8")).hexdigest()


def clip(words):
    s = " ".join(words)
    return s if len(s) <= MAX_SIDE else s[:MAX_SIDE] + " …"


def word_diff(text_a, text_b):
    """Changed spans as '… context [A: … | B: …] context …' strings."""
    wa, wb = text_a.split()[:MAX_WORDS], text_b.split()[:MAX_WORDS]
    ops = [op for op in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    hunks = []
    for _, i1, i2, j1, j2 in ops[:MAX_HUNKS]:
        before = " ".join(wa[max(0, i1 - CONTEXT):i1])
        after = " ".join(wa[i2:i2 + CONTEXT])
        a_part = clip(wa[i1:i2]) or "∅"
        b_part = clip(wb[j1:j2]) or "∅"
        hunks.append(f"… {before} [A: {a_part} | B: {b_part}] {after} …")
    if len(ops) > MAX_HUNKS:
        hunks.append(f"(+{len(ops) - MAX_HUNKS} more changes)")
    return hunks, len(wa), len(wb)


def draw(pairs, per_band, site_cap):
    """Seeded pick per (set, band): site-pair cap, each page once per band."""
    by_cell = defaultdict(list)
    for p in pairs:
        by_cell[(p["set"], band_of(p["jaccard"]))].append(p)
    picked = []
    for cell, cands in sorted(by_cell.items()):
        cands.sort(key=lambda p: seeded_order(p["url_a"] + p["url_b"]))
        sites, pages, n = Counter(), set(), 0
        for p in cands:
            key = tuple(sorted((p["site_a"], p["site_b"])))
            if n >= per_band or sites[key] >= site_cap or {p["url_a"], p["url_b"]} & pages:
                continue
            sites[key] += 1
            pages |= {p["url_a"], p["url_b"]}
            picked.append(p)
            n += 1
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--per-band", type=int, default=30)
    ap.add_argument("--site-cap", type=int, default=10)
    ap.add_argument("--min-band", type=float, default=0.7, help="lowest band drawn; the go rule is judged at >= 0.7")
    args = ap.parse_args()

    # labels already given, keyed by URL pair, so a rebuild keeps them
    old = {}
    if os.path.exists(BATCH):
        old = {frozenset((r["url_a"], r["url_b"])): r["label"]
               for r in load_jsonl(BATCH) if r.get("label") and not r.get("auto")}

    pairs = [p for p in load_jsonl(PAIRS) if band_of(p["jaccard"]) >= args.min_band]
    picked = draw(pairs, args.per_band, args.site_cap)
    random.Random(SEED).shuffle(picked)

    text = {r["url"]: r["text"] for path in SETS.values() for r in load_jsonl(path) if r.get("text")}
    batch, scores = [], []
    for i, p in enumerate(picked, 1):
        hunks, n_a, n_b = word_diff(text[p["url_a"]], text[p["url_b"]])
        # identical text needs no judgment: same by the labeling rules
        auto = not hunks
        label = "same" if auto else old.get(frozenset((p["url_a"], p["url_b"])), "")
        batch.append({"pair": i, "url_a": p["url_a"], "url_b": p["url_b"],
                      "start": " ".join(text[p["url_a"]].split()[:START_WORDS]),
                      "words_a": n_a, "words_b": n_b, "diff": hunks,
                      "label": label, "auto": auto})
        scores.append({"pair": i, "set": p["set"], "band": band_of(p["jaccard"]),
                       "jaccard": p["jaccard"], "minhash": p["minhash"],
                       "site_a": p["site_a"], "site_b": p["site_b"]})
    write_jsonl(BATCH, batch)
    write_jsonl(SCORES, scores)

    print(f"{len(batch)} pairs -> {BATCH} (scores -> {SCORES})")
    n_auto = sum(r["auto"] for r in batch)
    print(f"  {n_auto} identical-text pairs auto-labeled same; "
          f"{sum(not r['label'] for r in batch)} left to label")
    if old:
        kept = sum(1 for r in batch if r["label"] and not r["auto"])
        print(f"  {kept} of {len(old)} hand labels carried over")
    print()
    cells = Counter((s["set"], s["band"]) for s in scores)
    print(f"  {'set':<9}" + "".join(f"{f'{b:.1f}+':>7}" for b in BANDS))
    for name in SETS:
        print(f"  {name:<9}" + "".join(f"{cells[(name, b)]:>7}" for b in BANDS))


if __name__ == "__main__":
    main()

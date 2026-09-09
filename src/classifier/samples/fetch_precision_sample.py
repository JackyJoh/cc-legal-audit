"""
Builds the hand-labeling batch that measures classifier precision inside the
legal domains, drawn as three score strata.

A plain random sample of the pool would land too few pages above 0.85 to read
precision there, and that is where the operating threshold is likely to sit.
Sampling each stratum on its own fixes how many labels fall in each, so the
figure above 0.85 is precise no matter how the pool's scores skew.

The strata, and what each one answers:

  >= 0.85    precision at the likely operating threshold
  0.60-0.85  precision through the band underneath it, so the threshold can be
             moved down after the labels come back rather than fixed first
  < 0.60     what the threshold discards, and whether the legal pages among
             them are a random mix or all one document type. A threshold that
             drops opinions while keeping statutes narrows the corpus by
             document type, and topic entropy is what the study measures.

Each stratum is sampled at its own rate, so any precision figure spanning more
than one of them has to weight each stratum by its share of the pool rather
than its share of the labels. The pool counts that weighting needs are written
to the stats file. A figure read off a single stratum, precision at 0.85
included, needs no weighting at all.

Pages are located, fetched and extracted with fetch_warc_text.py's own
functions, so the text is prepared exactly the way the training text was.
Anything else is a different input and the scores would not mean the same
thing. Fetches are cached, so a rerun costs nothing for pages already held.

Scores go to their own file, keyed by URL. The labeling agent reads the batch
and never sees them, because agreement with the model is the one thing this
batch exists to test.

Reads:
  data/candidates/legal_pool.jsonl

Writes:
  data/candidates/legal_sample_batch.jsonl   URLs for the labeling agent
  data/candidates/legal_sample_scores.jsonl  score and stratum, keyed by URL
  data/candidates/legal_sample_stats.json    pool counts per stratum
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from fetch_warc_text import load_jsonl, resolve_pointers  # noqa: E402
from features import MODES, legal_probs, load_bundle  # noqa: E402

from fetch_flagged_urls import fetch_text, load_existing_urls  # noqa: E402

POOL_FILE     = "data/candidates/legal_pool.jsonl"
TEXT_CACHE    = "data/candidates/_legal_pool_text_cache.jsonl"
POINTER_CACHE = "data/candidates/_legal_pool_pointers_cache.jsonl"
OUTPUT_FILE   = "data/candidates/legal_sample_batch.jsonl"
SCORES_FILE   = "data/candidates/legal_sample_scores.jsonl"
STATS_FILE    = "data/candidates/legal_sample_stats.json"
BATCH_ID      = "legal-precision-v1"
SEED          = 42

# Every batch row ships with the label field already present and holding this,
# so labeling is overwriting a value rather than adding a key, and any row still
# carrying it is visibly unlabeled instead of quietly missing.
LABEL_PLACEHOLDER = "label_here"

# Cut points and how many labels each stratum gets. The high cut is the likely
# operating threshold; the low cut is far enough under it that the labels still
# cover the band a lower threshold would pull in.
STRATA = [
    ("high", 0.85, 1.01, 100),
    ("mid",  0.60, 0.85, 100),
    ("low",  0.00, 0.60,  50),
]

# URLs already labeled or already handed to the labeling agent, so no page is
# judged twice and no page arrives carrying an answer.
EXISTING = [
    "data/processed/labeled_urls.jsonl",
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
    "data/candidates/host_sample_batch.jsonl",
    "data/candidates/bill_sample_batch.jsonl",
    "data/candidates/flagged_sample_batch.jsonl",
]


def stratum_of(prob):
    for name, lo, hi, _ in STRATA:
        if lo <= prob < hi:
            return name
    return None


def take(urls, k):
    """A seeded draw of k from a stratum.

    Sorting before shuffling matters: the scored rows arrive in whatever order
    the cache happened to hold them, so without the sort the same seed would
    draw a different sample on a different run.
    """
    ordered = sorted(urls)
    random.Random(SEED).shuffle(ordered)
    return ordered[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--model", default="models/text_clf.joblib")
    ap.add_argument("--pool", default=POOL_FILE)
    ap.add_argument("--output", default=OUTPUT_FILE)
    ap.add_argument("--scores", default=SCORES_FILE)
    ap.add_argument("--stats", default=STATS_FILE)
    args = ap.parse_args()

    rows = [{"url": r["url"]} for r in load_jsonl(args.pool)]
    print(f"pool: {len(rows)} URLs")

    print("\n1. WARC pointers")
    pointers = resolve_pointers(rows, POINTER_CACHE)

    print("\n2. Page text")
    have = fetch_text(rows, pointers, TEXT_CACHE)

    print("\n3. Scoring")
    bundle = load_bundle(args.model)
    field = MODES[bundle["mode"]][2]
    drawn = [have[r["url"]] for r in rows if r["url"] in have]
    scored = [r for r in drawn if r.get(field)]
    print(f"  {len(scored)} of {len(rows)} pool pages have text")

    probs = legal_probs(bundle, [r[field] for r in scored])
    by_url = {r["url"]: float(p) for r, p in zip(scored, probs)}

    existing = load_existing_urls(EXISTING)
    eligible = {u: p for u, p in by_url.items() if u not in existing}
    print(f"  {len(by_url) - len(eligible)} already labeled or batched, dropped")

    print(f"\n{'stratum':>8} {'range':>12} {'in pool':>9} {'available':>10} {'taken':>7}")
    picked, stats = [], {"pool_scored": len(by_url), "strata": {}}
    for name, lo, hi, k in STRATA:
        in_pool = sum(1 for p in by_url.values() if lo <= p < hi)
        avail = [u for u, p in eligible.items() if lo <= p < hi]
        chosen = take(avail, k)
        picked.extend(chosen)
        stats["strata"][name] = {
            "low": lo, "high": hi, "target": k,
            "pool_count": in_pool, "available": len(avail), "sampled": len(chosen),
        }
        print(f"{name:>8} {f'{lo:.2f}-{hi:.2f}':>12} {in_pool:>9} "
              f"{len(avail):>10} {len(chosen):>7}")
        if len(chosen) < k:
            print(f"         short by {k - len(chosen)}, draw a larger pool "
                  f"with fetch_legal_pool.py --n")

    picked.sort()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for url in picked:
            f.write(json.dumps({"url": url, "hint": "legal_domain_sample",
                                "batch": BATCH_ID, "label": LABEL_PLACEHOLDER}) + "\n")
    with open(args.scores, "w", encoding="utf-8") as f:
        for url in picked:
            f.write(json.dumps({"url": url,
                                "prob_legal": round(by_url[url], 6),
                                "stratum": stratum_of(by_url[url])}) + "\n")
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{len(picked)} URLs written : {args.output}")
    print(f"scores held separately: {args.scores}")
    print(f"pool counts for weighting: {args.stats}")


if __name__ == "__main__":
    main()

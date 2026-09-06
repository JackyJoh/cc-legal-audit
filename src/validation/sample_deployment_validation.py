"""
Draws a blind sample of flagged pages for hand-labeling, to measure precision
on the real crawl distribution.

Usage:
  python src/validation/sample_deployment_validation.py \
      --model models/text_clf.joblib --pool data/processed/crawl_text.jsonl

Every precision figure the project reports so far comes from the label set,
which is about 24% legal by construction. The open crawl is nearer 0.1%.
Precision does not transfer across that gap, because a rare positive class
means far more chances to be wrong for each chance to be right, so a model
that looks accurate on the label set can be mostly wrong in deployment
without any of its held-out numbers changing. Recall and false-positive rate
do transfer; precision has to be measured where it will be used.

How it works: score the pool with a saved model, keep everything at or above
the threshold, and draw a uniform random sample of those to label by hand.
The draw is random rather than confidence-sorted, because the top of the
ranking is the easy part and would flatter the estimate.

Two output files, separate on purpose:
  <prefix>.jsonl         URL only. Fill in "your_label" with legal/non_legal.
  <prefix>_scores.jsonl  the same URLs with the model's probability, kept out
                         of the labeling file so the score cannot anchor the
                         judgment.

The flag rate printed at the end is worth reading before labeling anything.
If the model flags a far larger share of the pool than the base rate of legal
pages, precision is already bounded low and no amount of labeling will change
that.

Replaces the earlier pair of near-identical scripts (one per pool). The pool
is now an argument, and the model is loaded rather than refit, so the sample
can name the model that produced it.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "classifier"))
from features import MODES, legal_probs, load_bundle, read_jsonl  # noqa: E402

DEFAULT_EXCLUDE = [
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
    "data/candidates/host_sample_batch.jsonl",
]
DEFAULT_PREFIX = "data/validation/deployment_sample"
SEED = 42
CHUNK_SIZE = 20000


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True,
                    help="jsonl to scan. For a text model this must already "
                         "carry page text, so run src/corpus/fetch_warc_text.py "
                         "over the URL sample first.")
    ap.add_argument("--n", type=int, default=100,
                    help="how many flagged rows to draw for labeling")
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: the threshold recorded in the model")
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                    help="jsonl files whose URLs were already labeled, so the "
                         "sample stays out-of-sample")
    ap.add_argument("--out-prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    random.seed(args.seed)

    bundle = load_bundle(args.model)
    field = MODES[bundle["mode"]][2]
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]

    seen = set()
    for path in args.exclude:
        if os.path.exists(path):
            seen.update(r["url"] for r in read_jsonl(path))
    print(f"\nexcluding {len(seen)} already-labeled URLs")

    pool = [r for r in read_jsonl(args.pool)
            if r.get(field) and r.get("url") not in seen]
    if not pool:
        raise SystemExit(f"no usable rows in {args.pool} (need a '{field}' field)")
    print(f"scoring {len(pool)} rows from {args.pool}")

    flagged = []
    for start in range(0, len(pool), CHUNK_SIZE):
        chunk = pool[start:start + CHUNK_SIZE]
        probs = legal_probs(bundle, [r[field] for r in chunk])
        flagged += [(r["url"], float(p)) for r, p in zip(chunk, probs)
                    if p >= threshold]
        print(f"  {min(start + CHUNK_SIZE, len(pool))}/{len(pool)}, "
              f"{len(flagged)} flagged")

    rate = len(flagged) / len(pool)
    print(f"\nflagged {len(flagged)}/{len(pool)} at threshold {threshold} "
          f"({rate:.4%} of the pool)")
    if not flagged:
        raise SystemExit("nothing cleared the threshold, nothing to label")

    sample = random.sample(flagged, min(args.n, len(flagged)))
    random.shuffle(sample)

    blind = f"{args.out_prefix}.jsonl"
    scores = f"{args.out_prefix}_scores.jsonl"
    os.makedirs(os.path.dirname(blind) or ".", exist_ok=True)
    with open(blind, "w", encoding="utf-8") as f:
        for url, _ in sample:
            f.write(json.dumps({"url": url, "your_label": ""}) + "\n")
    with open(scores, "w", encoding="utf-8") as f:
        for url, p in sample:
            f.write(json.dumps({"url": url, "model_probability": round(p, 6),
                                "model": args.model,
                                "threshold": threshold}) + "\n")

    print(f"\ndrew {len(sample)} of {len(flagged)} flagged rows")
    print(f"label these       : {blind}")
    print(f"scores (don't peek): {scores}")


if __name__ == "__main__":
    main()

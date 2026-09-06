"""
Groups a model's flagged pages by publisher.

Usage:
  python src/validation/scan_domain_breakdown.py \
      --model models/text_clf.joblib --pool data/candidates/raw_pool.jsonl

Answers the question a precision number cannot: when the model says legal,
whose pages is it actually finding? A model can post good precision while
only ever firing on two or three sites, which would mean the "legal" bucket
the dedup study measures is really those sites rather than legal text in
general. That is a validity problem for the study, not a model-quality one,
so it needs to be visible separately from the score.

Reads its model from disk rather than fitting one, so the breakdown always
names the model that produced it.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "classifier"))
from features import (MODES, domain_of, legal_probs, load_bundle,  # noqa: E402
                      read_jsonl)

DEFAULT_POOLS = [
    "data/candidates/raw_pool.jsonl",
    "data/candidates/raw_pool_no_ca.jsonl",
]
DEFAULT_EXCLUDE = [
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
    "data/candidates/host_sample_batch.jsonl",
]
CHUNK_SIZE = 20000


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", nargs="*", default=DEFAULT_POOLS,
                    help="one or more jsonl files to scan")
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: the threshold recorded in the model")
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE)
    args = ap.parse_args()

    bundle = load_bundle(args.model)
    field = MODES[bundle["mode"]][2]
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]

    seen = set()
    for path in args.exclude:
        if os.path.exists(path):
            seen.update(r["url"] for r in read_jsonl(path))

    pool, urls_seen = [], set()
    for path in args.pool:
        if not os.path.exists(path):
            print(f"  skipping missing {path}")
            continue
        for r in read_jsonl(path):
            u = r.get("url")
            if not r.get(field) or u in seen or u in urls_seen:
                continue
            urls_seen.add(u)
            pool.append(r)
    if not pool:
        raise SystemExit(f"no usable rows (need a '{field}' field)")
    print(f"\nscanning {len(pool)} unseen rows at threshold {threshold}")

    by_domain = Counter()
    n_flagged = 0
    for start in range(0, len(pool), CHUNK_SIZE):
        chunk = pool[start:start + CHUNK_SIZE]
        probs = legal_probs(bundle, [r[field] for r in chunk])
        for r, p in zip(chunk, probs):
            if p >= threshold:
                by_domain[domain_of(r["url"])] += 1
                n_flagged += 1
        done = min(start + CHUNK_SIZE, len(pool))
        if done % 200000 == 0 or done == len(pool):
            print(f"  {done}/{len(pool)}, {n_flagged} flagged")

    print(f"\nflagged {n_flagged}/{len(pool)} ({n_flagged / len(pool):.4%}) "
          f"across {len(by_domain)} registered domains")
    if not n_flagged:
        return
    print(f"\n{'domain':<32} {'count':>7} {'% flagged':>10} {'cumulative':>11}")
    cum = 0
    for domain, count in by_domain.most_common():
        cum += count
        print(f"{domain:<32} {count:>7} {count / n_flagged:>9.1%} "
              f"{cum / n_flagged:>10.1%}")


if __name__ == "__main__":
    main()

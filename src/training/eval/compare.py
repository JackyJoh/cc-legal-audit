"""
Compares two models' saved scores: one row per score, with the gap (A - B) and its
standard error. A negative gap on a loss row means A did better; a positive gap on
a benchmark row means A did better.

Perplexity and rare-token rows redraw the same pages for both models, so the error
is on the gap itself. Benchmarks don't save per-question results, so their error
is sqrt(error A^2 + error B^2). A benchmark's score is acc_norm where the task has
it, otherwise acc. Benchmarks are skipped if either model has no saved results.

Runs in the eval environment (~/venvs/eval on HiPerGator), on CPU, from the repo
folder. Never runs a model.

Usage:
  python src/training/eval/compare.py --a 0.6_seed1_final --b 0.9_seed1_final [--limit 200]

Model names are the start of the file names in evals/. --limit reads the files
from a --limit run of evaluate.py.

Reads (data/source/training/evals/, from evaluate.py):
  <name>_perplexity_pages.npz   page totals
  <name>_benchmarks.json        benchmark scores

Outputs (data/source/training/evals/):
  compare_<a>_vs_<b>.json   the printed rows

Prints: the table.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from perplexity import N_BUCKETS, estimate, make_resamples

OUT = "data/source/training/evals"

# the scores before the buckets in estimate's array, in order
WHOLE_SCORES = [("overall", "loss"), ("overall", "perplexity"),
                ("general", "loss"), ("general", "perplexity"),
                ("legal", "loss"), ("legal", "perplexity")]


# rows for perplexity and rarity, each (test, group, metric, A, B, gap, error)
def page_rows(name_a, name_b, suffix):
    a = np.load(f"{OUT}/{name_a}_perplexity{suffix}_pages.npz")
    b = np.load(f"{OUT}/{name_b}_perplexity{suffix}_pages.npz")
    if not np.array_equal(a["ids"], b["ids"]):
        sys.exit(f"{name_a} and {name_b} weren't scored on the same pages")
    if not np.array_equal(a["bucket_sets"], b["bucket_sets"]):
        sys.exit(f"{name_a} and {name_b} have different bucket sets")

    loss_a, tokens_a, loss_b, tokens_b = a["loss"], a["tokens"], b["loss"], b["tokens"]
    picks, resamples = make_resamples(a["groups"])

    rows = []
    for s, name in enumerate(a["bucket_sets"].tolist()):
        value_a = estimate(loss_a[s], tokens_a[s], picks)
        value_b = estimate(loss_b[s], tokens_b[s], picks)
        gaps = np.stack([estimate(loss_a[s], tokens_a[s], r) - estimate(loss_b[s], tokens_b[s], r)
                         for r in resamples])
        error = gaps.std(axis=0, ddof=1)

        def row(test, group, metric, i):
            return (test, group, metric, float(value_a[i]), float(value_b[i]),
                    float(value_a[i] - value_b[i]), float(error[i]))

        # bucket sets only differ in their buckets, so the rest is the same in each
        if s == 0:
            rows += [row("perplexity", group, metric, i)
                     for i, (group, metric) in enumerate(WHOLE_SCORES)]
        if name != "none":
            rows += [row(f"rare_{name}", f"bucket_{b}", "loss", len(WHOLE_SCORES) + b)
                     for b in range(N_BUCKETS)]
    return rows


# rows for benchmarks, the same shape as page_rows'
def benchmark_rows(name_a, name_b, suffix):
    paths = [f"{OUT}/{name}_benchmarks{suffix}.json" for name in (name_a, name_b)]
    if not all(os.path.exists(p) for p in paths):
        print("skipping benchmarks: no saved results for both models")
        return []

    scores = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            scores.append(json.load(f)["scores"])

    rows = []
    for task, metrics in scores[0].items():
        metric = "acc_norm" if "acc_norm" in metrics else "acc"
        value_a, error_a = metrics[metric]
        value_b, error_b = scores[1][task][metric]
        # a stderr can be "N/A" in lm-eval's results
        error = math.hypot(error_a, error_b) if isinstance(error_a, float) \
            and isinstance(error_b, float) else "N/A"
        rows.append(("benchmarks", task, metric, value_a, value_b, value_a - value_b, error))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="model name, as in evals/ file names")
    ap.add_argument("--b", required=True, help="model name, as in evals/ file names")
    ap.add_argument("--limit", type=int, help="read files from a --limit run")
    args = ap.parse_args()

    suffix = f"_limit{args.limit}" if args.limit else ""
    rows = page_rows(args.a, args.b, suffix) + benchmark_rows(args.a, args.b, suffix)

    print(f"\n{args.a} vs {args.b}\n")
    print(f"{'test':12} {'group':16} {'metric':12} {'A':>9} {'B':>9} {'gap':>9} {'+/-':>8}")
    for test, group, metric, value_a, value_b, gap, error in rows:
        error_text = f"{error:.4f}" if isinstance(error, float) else error
        print(f"{test:12} {group:16} {metric:12} {value_a:9.4f} {value_b:9.4f} "
              f"{gap:9.4f} {error_text:>8}")

    out = f"{OUT}/compare_{args.a}_vs_{args.b}{suffix}.json"
    columns = ("test", "group", "metric", "a", "b", "gap", "error")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"a": args.a, "b": args.b, "limit": args.limit,
                   "rows": [dict(zip(columns, row)) for row in rows]}, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

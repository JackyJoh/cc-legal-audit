"""
Prints the scores saved by evaluate.py as one table: a row per group and metric,
a column per model.

Usage:
  python src/training/eval/results.py [--test benchmarks]
"""
import argparse
import glob
import json
import os

OUT = "data/source/training/evals"

ap = argparse.ArgumentParser()
ap.add_argument("--test", default="benchmarks")
args = ap.parse_args()

# column name -> scores, from one file per model
models = {}
for path in sorted(glob.glob(f"{OUT}/*_{args.test}*.json")):
    name = os.path.basename(path).removesuffix(".json").replace(f"_{args.test}", "")
    with open(path, encoding="utf-8") as f:
        models[name] = json.load(f)["scores"]

if not models:
    raise SystemExit(f"no {args.test} results in {OUT}")

rows = []
for scores in models.values():
    for group, metrics in scores.items():
        for metric in metrics:
            if (group, metric) not in rows:
                rows.append((group, metric))


def cell(name, group, metric):
    score = models[name].get(group, {}).get(metric)
    if score is None:
        return "-"
    value, stderr = score
    error = f"{stderr:.3f}" if isinstance(stderr, float) else stderr
    return f"{value:.3f} +/- {error}"


widths = {name: max(len(name), 15) for name in models}
print(f"{'group':16} {'metric':12} " + " ".join(f"{n:{widths[n]}}" for n in models))
for group, metric in rows:
    cells = " ".join(f"{cell(n, group, metric):{widths[n]}}" for n in models)
    print(f"{group:16} {metric:12} {cells}")

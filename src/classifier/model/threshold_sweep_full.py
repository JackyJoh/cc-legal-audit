"""
Writes the full threshold sweep to CSV for the writeup.

Usage: python src/classifier/model/threshold_sweep_full.py --features url|text

Same split and same feature recipe as train_classifier.py, at every threshold
from 0.05 to 0.95 instead of the handful the trainer prints, with raw tp/fp/fn
counts alongside the rates so the numbers can be recomputed from the file.

Publishers appear on both sides of this split, so these are within-publisher
numbers. eval_grouped.py is what measures performance on unseen publishers.
"""
import argparse
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from features import MODES, load_labeled
from train_classifier import SEED, TEST_SIZE, fit, legal_probs

OUTPUT_CSV = "data/processed/threshold_sweep_{mode}.csv"
THRESHOLDS = [round(t * 0.05, 2) for t in range(1, 20)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--features", choices=sorted(MODES), default="text")
    ap.add_argument("--output", default=None,
                    help="default: data/processed/threshold_sweep_<features>.csv")
    args = ap.parse_args()

    mode = args.features
    out_csv = args.output or OUTPUT_CSV.format(mode=mode)

    urls, docs, labels, domains, path = load_labeled(mode)
    print(f"{len(urls)} labeled rows from {path} "
          f"({labels.count('legal')} legal, {labels.count('non_legal')} non_legal)")

    X_tr, X_te, y_tr, y_te, d_tr, _ = train_test_split(
        docs, labels, domains, test_size=TEST_SIZE, random_state=SEED,
        stratify=labels)
    vec, clf = fit(mode, X_tr, y_tr, d_tr)
    probs = legal_probs(vec, clf, X_te)
    y_bin = [1 if y == "legal" else 0 for y in y_te]
    print(f"train {len(X_tr)}  test {len(X_te)} "
          f"({sum(y_bin)} legal / {len(y_bin) - sum(y_bin)} non_legal)")

    rows = []
    for t in THRESHOLDS:
        preds = [1 if p >= t else 0 for p in probs]
        rows.append({
            "threshold": t,
            "precision": round(precision_score(y_bin, preds, zero_division=0), 4),
            "recall": round(recall_score(y_bin, preds, zero_division=0), 4),
            "f1": round(f1_score(y_bin, preds, zero_division=0), 4),
            "n_flagged": sum(preds),
            "tp": sum(1 for pr, yt in zip(preds, y_bin) if pr and yt),
            "fp": sum(1 for pr, yt in zip(preds, y_bin) if pr and not yt),
            "fn": sum(1 for pr, yt in zip(preds, y_bin) if not pr and yt),
        })

    print(f"\n{'thresh':>7} {'precision':>10} {'recall':>8} {'f1':>7} "
          f"{'flagged':>8} {'tp':>4} {'fp':>4} {'fn':>4}")
    for r in rows:
        print(f"{r['threshold']:>7.2f} {r['precision']:>10.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>7.4f} {r['n_flagged']:>8} {r['tp']:>4} {r['fp']:>4} "
              f"{r['fn']:>4}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()

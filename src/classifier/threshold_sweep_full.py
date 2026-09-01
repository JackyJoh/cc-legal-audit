"""
Full precision/recall/F1 sweep across confidence thresholds for the current
classifier, on the same held-out test split train_classifier.py uses.
Written for the record (research writeup), not just eyeballing.
"""
import csv
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

LABELED_FILES = ["data/processed/labeled_urls.jsonl"]
SEED = 42
OUTPUT_CSV = "data/processed/threshold_sweep_results.csv"


def load_labeled(paths):
    by_url = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                by_url.setdefault(obj["url"], obj["label"])
    urls = list(by_url.keys())
    labels = list(by_url.values())
    return urls, labels


def main():
    urls, labels = load_labeled(LABELED_FILES)
    print(f"Loaded {len(urls)} labeled URLs "
          f"({labels.count('legal')} legal, {labels.count('non_legal')} non_legal)")

    X_train, X_test, y_train, y_test = train_test_split(
        urls, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)} "
          f"({y_test.count('legal')} legal, {y_test.count('non_legal')} non_legal)")

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X_train_vec, y_train)

    legal_idx = list(clf.classes_).index("legal")
    test_probs = clf.predict_proba(X_test_vec)[:, legal_idx]
    y_test_bin = [1 if y == "legal" else 0 for y in y_test]

    thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 .. 0.95

    rows = []
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in test_probs]
        n_flagged = sum(preds)
        tp = sum(1 for pr, yt in zip(preds, y_test_bin) if pr == 1 and yt == 1)
        fp = sum(1 for pr, yt in zip(preds, y_test_bin) if pr == 1 and yt == 0)
        fn = sum(1 for pr, yt in zip(preds, y_test_bin) if pr == 0 and yt == 1)
        p = precision_score(y_test_bin, preds, zero_division=0)
        r = recall_score(y_test_bin, preds, zero_division=0)
        f1 = f1_score(y_test_bin, preds, zero_division=0)
        rows.append({
            "threshold": t, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "n_flagged": n_flagged, "tp": tp, "fp": fp, "fn": fn,
        })

    print(f"\n{'thresh':>7} {'precision':>10} {'recall':>8} {'f1':>7} "
          f"{'flagged':>8} {'tp':>4} {'fp':>4} {'fn':>4}")
    for row in rows:
        print(f"{row['threshold']:>7.2f} {row['precision']:>10.4f} {row['recall']:>8.4f} "
              f"{row['f1']:>7.4f} {row['n_flagged']:>8} {row['tp']:>4} {row['fp']:>4} {row['fn']:>4}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "threshold", "precision", "recall", "f1", "n_flagged", "tp", "fp", "fn"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nHeld-out test set: {len(X_test)} total "
          f"({sum(y_test_bin)} legal / {len(y_test_bin) - sum(y_test_bin)} non_legal)")
    print(f"Written: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

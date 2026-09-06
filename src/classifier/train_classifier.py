"""
Trains the legal/non-legal classifier and writes it to disk so it can score
documents later.

Usage: python src/classifier/train_classifier.py --features url|text

Feature recipes come from features.py rather than being spelled out here, so
the trainer and eval_grouped.py cannot end up fitting different models.

Two fits per run, on purpose:

  1. On an 80/20 stratified split, to report held-out metrics and a threshold
     sweep. These are the numbers to quote.
  2. On all the labels, and that is the model saved to disk. Refitting on
     everything is the normal thing to ship, but it means the saved model has
     seen every row, so its own predictions on the label set are worthless as
     a measurement. That is what fit (1) is for.

The saved bundle carries the fitted vectorizer, the fitted model, and enough
provenance to tell later whether a score came from this model: which label
file was used and its hash, which extractor produced the training text, and
the library versions. score.py refuses to run if the vectorizer and the
documents it is handed disagree about any of that.

Output: models/<mode>_clf.joblib, gitignored. It rebuilds from the label
files in about 5 seconds, so there is no reason to version it.
"""
import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from features import (MODES, MIN_DOMAIN_DF, load_labeled, make_classifier,
                      read_jsonl)

SEED = 42
TEST_SIZE = 0.2
MODEL_DIR = "models"
THRESHOLDS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95]

# Provisional. Recorded in the bundle as a default, not as a validated
# operating point: both were picked from held-out label-set metrics, where
# legal is ~24% of rows. In the real crawl it is nearer 0.1%, and precision
# does not survive that change of base rate. Until the deployment sample is
# labeled, treat these as placeholders.
DEFAULT_THRESHOLD = {"url": 0.85, "text": 0.60}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extractors_used(mode, path):
    """Which text extractor produced the training documents.

    Only meaningful in text mode. trafilatura 1.x and 2.x return different
    text for the same HTML, so a model trained on one and scoring the other
    is not doing what it looks like it is doing.
    """
    if mode != "text":
        return []
    return sorted({r.get("extractor") for r in read_jsonl(path)
                   if r.get("text") and r.get("extractor")})


def fit(mode, docs, labels, domains):
    make_vec, _, _ = MODES[mode]
    vec = make_vec()
    Xv = vec.fit_transform(docs, domains)
    clf = make_classifier()
    clf.fit(Xv, labels)
    return vec, clf


def legal_probs(vec, clf, docs):
    idx = list(clf.classes_).index("legal")
    return clf.predict_proba(vec.transform(docs))[:, idx]


def report_holdout(mode, docs, labels, domains):
    """Fit on 80%, score the held-out 20%. Publishers appear on both sides of
    this split, so it overstates performance on unseen sites. eval_grouped.py
    is the script that measures that gap."""
    X_tr, X_te, y_tr, y_te, d_tr, _ = train_test_split(
        docs, labels, domains, test_size=TEST_SIZE, random_state=SEED,
        stratify=labels)
    vec, clf = fit(mode, X_tr, y_tr, d_tr)
    probs = legal_probs(vec, clf, X_te)
    y_bin = [1 if y == "legal" else 0 for y in y_te]

    print(f"\n--- held-out test set ({len(X_te)} rows: {sum(y_bin)} legal / "
          f"{len(y_bin) - sum(y_bin)} non_legal) ---")
    print(classification_report(y_te, clf.predict(vec.transform(X_te))))

    print("--- threshold sweep (held-out) ---")
    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'f1':>7} {'flagged':>8}")
    for t in THRESHOLDS:
        preds = [1 if p >= t else 0 for p in probs]
        print(f"{t:>7.2f} "
              f"{precision_score(y_bin, preds, zero_division=0):>10.3f} "
              f"{recall_score(y_bin, preds, zero_division=0):>8.3f} "
              f"{f1_score(y_bin, preds, zero_division=0):>7.3f} "
              f"{sum(preds):>8}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", choices=sorted(MODES), default="text",
                    help="what the model reads: the URL string, or the "
                         "extracted page text")
    ap.add_argument("--threshold", type=float, default=None,
                    help="operating threshold recorded in the bundle "
                         "(default: 0.85 for url, 0.60 for text, both "
                         "provisional pending the deployment sample)")
    ap.add_argument("--out", default=None,
                    help="where to write the bundle "
                         "(default: models/<features>_clf.joblib)")
    args = ap.parse_args()

    mode = args.features
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD[mode]
    out_path = args.out or os.path.join(MODEL_DIR, f"{mode}_clf.joblib")

    print(f"--- loading (features: {mode}) ---")
    urls, docs, labels, domains, path = load_labeled(mode)
    print(f"{len(urls)} labeled rows from {path} "
          f"({labels.count('legal')} legal, {labels.count('non_legal')} non_legal) "
          f"across {len(set(domains))} registered domains")

    report_holdout(mode, docs, labels, domains)

    print(f"\n--- refitting on all {len(docs)} rows for the saved model ---")
    vec, clf = fit(mode, docs, labels, domains)
    n_features = len(vec.get_feature_names_out())
    print(f"vocabulary: {n_features} features after the domain purity filter "
          f"(min_domain_df={MIN_DOMAIN_DF})")

    bundle = {
        "vectorizer": vec,
        "classifier": clf,
        "mode": mode,
        "threshold": threshold,
        "meta": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label_file": path,
            "label_file_sha256": file_sha256(path),
            "n_train": len(docs),
            "n_legal": labels.count("legal"),
            "n_domains": len(set(domains)),
            "n_features": n_features,
            "min_domain_df": MIN_DOMAIN_DF,
            "extractors": extractors_used(mode, path),
            "seed": SEED,
            "sklearn": sklearn.__version__,
            "python": platform.python_version(),
        },
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, out_path, compress=3)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"\nwrote {out_path} ({size_mb:.1f} MB), threshold {threshold}")
    print(json.dumps(bundle["meta"], indent=2))


if __name__ == "__main__":
    main()

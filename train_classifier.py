"""
Temp trainer: char n-gram TF-IDF + logistic regression on the URL string
alone. Trains on data/processed/labeled_urls.jsonl, reports held-out
metrics, then scores a batch of genuinely out-of-sample URLs (pulled from
raw_pool.jsonl, filtered to URLs never in candidates.jsonl, so the model
has never seen them in any form) and prints the ones it's most confident
are legal.
"""
import json
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

LABELED_FILE = "data/processed/labeled_urls.jsonl"
CANDIDATES_FILE = "data/candidates/candidates.jsonl"
RAW_POOL_FILE = "data/candidates/raw_pool.jsonl"
SEED = 42
N_OOS = 20
# precision over recall on purpose: false positives pollute the (tiny) legal
# bucket that topic diversity gets measured on, false negatives just get
# reabsorbed into the (huge) non_legal bucket where they're a rounding error
OPERATING_THRESHOLD = 0.9


def load_labeled(path):
    urls, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            urls.append(obj["url"])
            labels.append(obj["label"])
    return urls, labels


def load_urls(path):
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                urls.append(json.loads(line)["url"])
    return urls


def main():
    random.seed(SEED)

    urls, labels = load_labeled(LABELED_FILE)
    print(f"Loaded {len(urls)} labeled URLs "
          f"({labels.count('legal')} legal, {labels.count('non_legal')} non_legal)")

    X_train, X_test, y_train, y_test = train_test_split(
        urls, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X_train_vec, y_train)

    print("\n--- held-out test set (default 0.5 threshold) ---")
    print(classification_report(y_test, clf.predict(X_test_vec)))

    legal_idx = list(clf.classes_).index("legal")
    test_probs = clf.predict_proba(X_test_vec)[:, legal_idx]
    y_test_bin = [1 if y == "legal" else 0 for y in y_test]

    print("--- threshold sweep on legal confidence (held-out test set) ---")
    print(f"{'threshold':>9} {'precision':>9} {'recall':>9} {'f1':>9} {'n_flagged':>9}")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        preds = [1 if p >= t else 0 for p in test_probs]
        p = precision_score(y_test_bin, preds, zero_division=0)
        r = recall_score(y_test_bin, preds, zero_division=0)
        f1 = f1_score(y_test_bin, preds, zero_division=0)
        n_flagged = sum(preds)
        print(f"{t:>9.2f} {p:>9.3f} {r:>9.3f} {f1:>9.3f} {n_flagged:>9}")

    # genuinely unseen URLs: raw pool minus anything ever in the candidate batch
    candidates = set(load_urls(CANDIDATES_FILE))
    raw_pool = load_urls(RAW_POOL_FILE)
    oos_pool = [u for u in raw_pool if u not in candidates]
    oos_sample = random.sample(oos_pool, min(2000, len(oos_pool)))

    oos_vec = vectorizer.transform(oos_sample)
    probs = clf.predict_proba(oos_vec)
    legal_idx = list(clf.classes_).index("legal")
    scored = sorted(zip(oos_sample, probs[:, legal_idx]), key=lambda x: -x[1])

    print(f"\n--- top {N_OOS} most-confident 'legal' predictions out of "
          f"{len(oos_sample)} unseen raw-pool URLs ---")
    for url, p in scored[:N_OOS]:
        print(f"{p:.3f}  {url}")

    n_at_threshold = sum(1 for _, p in scored if p >= OPERATING_THRESHOLD)
    print(f"\nAt operating threshold {OPERATING_THRESHOLD}: "
          f"{n_at_threshold}/{len(oos_sample)} unseen URLs would be kept as legal")

    # scan the full 600k raw pool in order, in chunks, stopping as soon as
    # N_TARGET_HITS clear the operating threshold
    N_TARGET_HITS = 5
    CHUNK_SIZE = 5000
    print(f"\n--- scanning full raw pool ({len(raw_pool)} URLs) for "
          f"{N_TARGET_HITS} hits at threshold {OPERATING_THRESHOLD} ---")
    hits = []
    scanned = 0
    for start in range(0, len(raw_pool), CHUNK_SIZE):
        chunk = raw_pool[start:start + CHUNK_SIZE]
        chunk_vec = vectorizer.transform(chunk)
        chunk_probs = clf.predict_proba(chunk_vec)[:, legal_idx]
        scanned += len(chunk)
        for url, p in zip(chunk, chunk_probs):
            if p >= OPERATING_THRESHOLD:
                hits.append((url, p))
                print(f"  [{scanned} scanned] {p:.3f}  {url}")
                if len(hits) >= N_TARGET_HITS:
                    break
        if len(hits) >= N_TARGET_HITS:
            break

    print(f"\nFound {len(hits)} hits after scanning {scanned}/{len(raw_pool)} URLs.")


if __name__ == "__main__":
    main()

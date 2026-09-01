"""
Second deployment-validation sample: same idea as
sample_deployment_validation.py, but scoring the fresh non-.ca pool
(raw_pool_no_ca.jsonl) instead of the original raw_pool.jsonl, so this
sample can't overlap with the first 21-URL sample and can't be Canadian.
"""
import json
import os
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELED_FILES = ["data/processed/labeled_urls.jsonl"]
CANDIDATES_FILES = [
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
    "data/candidates/host_sample_batch.jsonl",
]
RAW_POOL_FILE = "data/candidates/raw_pool_no_ca.jsonl"
SEED = 42
N_SAMPLE = 20
OPERATING_THRESHOLD = 0.85
CHUNK_SIZE = 20000

BLIND_OUTPUT = "data/validation/deployment_sample_2.jsonl"
SCORES_OUTPUT = "data/validation/deployment_sample_2_scores.jsonl"


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
    return list(by_url.keys()), list(by_url.values())


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

    urls, labels = load_labeled(LABELED_FILES)
    print(f"Training on {len(urls)} labeled URLs "
          f"({labels.count('legal')} legal, {labels.count('non_legal')} non_legal)")

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    X = vectorizer.fit_transform(urls)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X, labels)
    legal_idx = list(clf.classes_).index("legal")

    already_used = set()
    for path in CANDIDATES_FILES:
        already_used.update(load_urls(path))

    raw_pool = load_urls(RAW_POOL_FILE)
    unseen_pool = [u for u in raw_pool if u not in already_used]
    print(f"Raw pool (no .ca): {len(raw_pool)}  Unseen: {len(unseen_pool)}")

    flagged = []
    for start in range(0, len(unseen_pool), CHUNK_SIZE):
        chunk = unseen_pool[start:start + CHUNK_SIZE]
        probs = clf.predict_proba(vectorizer.transform(chunk))[:, legal_idx]
        for url, p in zip(chunk, probs):
            if p >= OPERATING_THRESHOLD:
                flagged.append((url, float(p)))

    print(f"Total flagged at threshold {OPERATING_THRESHOLD}: {len(flagged)}")
    for url, p in sorted(flagged, key=lambda x: -x[1]):
        print(f"  {p:.3f}  {url}")

    sample = random.sample(flagged, min(N_SAMPLE, len(flagged)))
    random.shuffle(sample)

    os.makedirs(os.path.dirname(BLIND_OUTPUT), exist_ok=True)
    with open(BLIND_OUTPUT, "w", encoding="utf-8") as f:
        for url, _ in sample:
            f.write(json.dumps({"url": url, "your_label": ""}) + "\n")
    with open(SCORES_OUTPUT, "w", encoding="utf-8") as f:
        for url, p in sample:
            f.write(json.dumps({"url": url, "model_probability": p}) + "\n")

    print(f"\nSampled {len(sample)} URLs for hand labeling.")
    print(f"Blind review file: {BLIND_OUTPUT}")


if __name__ == "__main__":
    main()

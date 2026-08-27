"""
Deployment-distribution precision validation, step 1: source the sample.

Everything measured so far (91.4% precision @ 0.85) comes from a curated
batch that's ~30% legal by construction. The real crawl is ~0.07-0.14%
legal, and precision is base-rate sensitive, so that number isn't
necessarily what you'd see in production. This scores the full raw pool
with the current model, takes every URL that clears OPERATING_THRESHOLD,
and draws a genuinely random (not confidence-sorted) sample of N_SAMPLE of
them for hand labeling - that gives a real precision estimate under the
actual deployment distribution.

Two output files, kept separate on purpose:
  - data/validation/deployment_sample.jsonl: url only, blind, for hand
    labeling. Fill in "your_label" as "legal" or "non_legal" per URL.
  - data/validation/deployment_sample_scores.jsonl: same URLs plus the
    model's predicted probability, kept out of the blind file so a a
    priori confidence doesn't anchor the manual judgment.

Run compute_deployment_precision.py after hand-labeling to get the
final number.
"""
import json
import os
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  # noqa: F401 (kept consistent w/ train_classifier.py imports)

LABELED_FILES = [
    "data/processed/labeled_urls.jsonl",
    "data/processed/targeted_labeled_urls.jsonl",
]
CANDIDATES_FILES = [
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
]
RAW_POOL_FILE = "data/candidates/raw_pool.jsonl"
SEED = 42
N_SAMPLE = 100
OPERATING_THRESHOLD = 0.85
CHUNK_SIZE = 20000

BLIND_OUTPUT = "data/validation/deployment_sample.jsonl"
SCORES_OUTPUT = "data/validation/deployment_sample_scores.jsonl"


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
    print(f"Raw pool: {len(raw_pool)}  Unseen (never trained/labeled on): {len(unseen_pool)}")

    flagged = []
    for start in range(0, len(unseen_pool), CHUNK_SIZE):
        chunk = unseen_pool[start:start + CHUNK_SIZE]
        probs = clf.predict_proba(vectorizer.transform(chunk))[:, legal_idx]
        for url, p in zip(chunk, probs):
            if p >= OPERATING_THRESHOLD:
                flagged.append((url, float(p)))
        print(f"  scanned {min(start + CHUNK_SIZE, len(unseen_pool))}/{len(unseen_pool)}, "
              f"{len(flagged)} flagged so far")

    print(f"\nTotal flagged at threshold {OPERATING_THRESHOLD}: {len(flagged)}")

    sample = random.sample(flagged, min(N_SAMPLE, len(flagged)))
    random.shuffle(sample)  # sampling order != display order, belt and suspenders

    os.makedirs(os.path.dirname(BLIND_OUTPUT), exist_ok=True)
    with open(BLIND_OUTPUT, "w", encoding="utf-8") as f:
        for url, _ in sample:
            f.write(json.dumps({"url": url, "your_label": ""}) + "\n")
    with open(SCORES_OUTPUT, "w", encoding="utf-8") as f:
        for url, p in sample:
            f.write(json.dumps({"url": url, "model_probability": p}) + "\n")

    print(f"\nSampled {len(sample)} URLs for hand labeling.")
    print(f"Blind review file (label these): {BLIND_OUTPUT}")
    print(f"Score reference (don't peek until done): {SCORES_OUTPUT}")


if __name__ == "__main__":
    main()

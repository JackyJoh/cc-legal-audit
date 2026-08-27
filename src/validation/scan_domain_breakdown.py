"""
Large-scale scan across both raw pools (raw_pool.jsonl, 600k, unrestricted;
raw_pool_no_ca.jsonl, 1.5M, .ca excluded) with the current classifier at
OPERATING_THRESHOLD, grouped by root domain. Answers: when the model says
"legal" with high confidence, which domains is it actually finding?
"""
import json
from collections import Counter
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELED_FILES = [
    "data/processed/labeled_urls.jsonl",
    "data/processed/targeted_labeled_urls.jsonl",
]
CANDIDATES_FILES = [
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
]
RAW_POOL_FILES = [
    "data/candidates/raw_pool.jsonl",
    "data/candidates/raw_pool_no_ca.jsonl",
]
OPERATING_THRESHOLD = 0.85
CHUNK_SIZE = 20000


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


def root_domain(url):
    host = urlparse(url).netloc.split(":")[0]
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def main():
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

    seen = set()
    pool = []
    for path in RAW_POOL_FILES:
        for u in load_urls(path):
            if u in already_used or u in seen:
                continue
            seen.add(u)
            pool.append(u)

    print(f"Combined unseen pool: {len(pool)} URLs")

    flagged_by_domain = Counter()
    total_scanned = 0
    for start in range(0, len(pool), CHUNK_SIZE):
        chunk = pool[start:start + CHUNK_SIZE]
        probs = clf.predict_proba(vectorizer.transform(chunk))[:, legal_idx]
        for url, p in zip(chunk, probs):
            if p >= OPERATING_THRESHOLD:
                flagged_by_domain[root_domain(url)] += 1
        total_scanned += len(chunk)
        if total_scanned % 200000 == 0 or total_scanned == len(pool):
            print(f"  scanned {total_scanned}/{len(pool)}, "
                  f"{sum(flagged_by_domain.values())} flagged so far")

    total_flagged = sum(flagged_by_domain.values())
    print(f"\nTotal scanned: {total_scanned}")
    print(f"Total flagged at threshold {OPERATING_THRESHOLD}: {total_flagged}")
    print(f"\n{'domain':<25} {'count':>7} {'% of flagged':>13}")
    for domain, count in flagged_by_domain.most_common():
        pct = count / total_flagged * 100
        print(f"{domain:<25} {count:>7} {pct:>12.1f}%")


if __name__ == "__main__":
    main()

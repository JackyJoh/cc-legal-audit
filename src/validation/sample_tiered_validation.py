"""
Turns data/candidates/tier_validation_pool.jsonl (fetch_tier_urls.py's
output) into the blind sample for the two-tier deployment validation:
top-3 domains scored at 0.85, everything else at 0.70.

Scores every pooled URL, keeps what clears its tier's threshold, then draws
a genuinely random (not confidence-sorted) sample per tier up to
N_PER_TIER, capped at PER_DOMAIN_CAP per registered domain so one deep
domain (cornell.edu, or whichever minor domain turns out to have the most
eligible captures) can't crowd out the rest of its tier. Capping is by
registered domain, not hostname, since several hostnames in the minor tier
(e.g. justice.gc.ca's six mirror spellings) collapse to one domain and
should be capped together.

Two output files per tier, same pattern as sample_deployment_validation.py:
  - data/validation/tiered_sample_<tier>.jsonl: url only, blind, for hand
    labeling. Fill in "your_label" as "legal" or "non_legal" per URL.
  - data/validation/tiered_sample_<tier>_scores.jsonl: same URLs plus the
    model's predicted probability and registered domain, kept out of the
    blind file.
"""
import json
import os
import random
from collections import defaultdict

import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELED_FILES = ["data/processed/labeled_urls.jsonl"]
POOL_FILE = "data/candidates/tier_validation_pool.jsonl"
SEED = 44
N_PER_TIER = 100
PER_DOMAIN_CAP = {"top3": 40, "minor": 25}
THRESHOLDS = {"top3": 0.85, "minor": 0.70}

_extract = tldextract.TLDExtract(suffix_list_urls=())
def registered_domain(host):
    return _extract(host).top_domain_under_public_suffix


def load_labeled(paths):
    by_url = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    by_url.setdefault(obj["url"], obj["label"])
    return list(by_url.keys()), list(by_url.values())


def load_pool(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def capped_sample(flagged, n_target, per_domain_cap, rng):
    """Random draw up to n_target, capped per registered domain. flagged is
    a list of (url, prob, domain) tuples."""
    order = flagged[:]
    rng.shuffle(order)
    taken = []
    domain_counts = defaultdict(int)
    for url, prob, domain in order:
        if len(taken) >= n_target:
            break
        if domain_counts[domain] >= per_domain_cap:
            continue
        taken.append((url, prob, domain))
        domain_counts[domain] += 1
    return taken, domain_counts


def main():
    rng = random.Random(SEED)

    urls, labels = load_labeled(LABELED_FILES)
    print(f"Training on {len(urls)} labeled URLs "
          f"({labels.count('legal')} legal, {labels.count('non_legal')} non_legal)")

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    X = vectorizer.fit_transform(urls)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(X, labels)
    legal_idx = list(clf.classes_).index("legal")

    pool = load_pool(POOL_FILE)
    print(f"Pool: {len(pool)} URLs")

    pool_urls = [r["url"] for r in pool]
    probs = clf.predict_proba(vectorizer.transform(pool_urls))[:, legal_idx]

    flagged_by_tier = defaultdict(list)
    for row, p in zip(pool, probs):
        tier = row["tier"]
        if p >= THRESHOLDS[tier]:
            domain = registered_domain(row["host"])
            flagged_by_tier[tier].append((row["url"], float(p), domain))

    for tier in ("top3", "minor"):
        flagged = flagged_by_tier[tier]
        n_domains = len({d for _, _, d in flagged})
        print(f"\n--- {tier} tier (threshold {THRESHOLDS[tier]}) ---")
        print(f"Flagged in pool: {len(flagged)} across {n_domains} registered domains")

        sample, domain_counts = capped_sample(
            flagged, N_PER_TIER, PER_DOMAIN_CAP[tier], rng)
        print(f"Sampled: {len(sample)} (target {N_PER_TIER}, cap {PER_DOMAIN_CAP[tier]}/domain)")
        for d, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
            print(f"  {n:>4}  {d}")

        blind_path = f"data/validation/tiered_sample_{tier}.jsonl"
        scores_path = f"data/validation/tiered_sample_{tier}_scores.jsonl"
        os.makedirs(os.path.dirname(blind_path), exist_ok=True)

        shuffled = sample[:]
        rng.shuffle(shuffled)
        with open(blind_path, "w", encoding="utf-8") as f:
            for url, _, _ in shuffled:
                f.write(json.dumps({"url": url, "your_label": ""}) + "\n")
        with open(scores_path, "w", encoding="utf-8") as f:
            for url, prob, domain in shuffled:
                f.write(json.dumps({"url": url, "model_probability": prob,
                                     "registered_domain": domain}) + "\n")

        print(f"Blind review file: {blind_path}")


if __name__ == "__main__":
    main()

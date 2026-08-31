"""
Grouped evaluation: does the classifier generalise to legal publishers it has
never seen, or is it recognising hostnames?

train_classifier.py splits randomly over URL rows. With a positive class
concentrated in a handful of registered domains, a random split puts the same
publisher's URLs on both sides of it, so char n-gram TF-IDF can post a strong
precision number by learning that publisher's own substrings. That measures
within-domain interpolation, not generalisation - and the deployment
validation points the same way, since its samples landed on the same domains
that dominate training.

The concentration table this script prints first is the evidence for that
claim; read it rather than any figure quoted in a comment.

This script runs three things:

1. The random-row split, to reproduce the current headline numbers.
2. A grouped split, holding whole registered domains out. Same model, same
   threshold sweep. The gap between (1) and (2) is the leakage.
3. Leave-one-domain-out: retrain without each legal-bearing domain in turn and
   score only that domain. This is the per-publisher version of the same
   question, and it says directly whether the thin domains recover.

Plus a coefficient audit - if the top positive features are substrings of
training hostnames, the per-host cap in sampling was too loose.

Grouping is by registered domain, not hostname: law.cornell.edu and
www.law.cornell.edu are the same publisher, and holding out only one of them
would leak the other.

Read this as a diagnostic, not as the shipped model's metrics.
"""
import json
import os
from collections import Counter
from urllib.parse import urlparse

import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

LABELED_FILES = [
    "data/processed/labeled_urls.jsonl",
    "data/processed/targeted_labeled_urls.jsonl",
    # written once the host-sample batch has been labeled; skipped if absent
    "data/processed/host_sample_labeled_urls.jsonl",
]
SEED = 42
OPERATING_THRESHOLD = 0.85
THRESHOLDS = [0.5, 0.65, 0.75, 0.85, 0.9]
# below this a domain's leave-one-out recall is too noisy to read
MIN_LEGAL_FOR_LODO = 5
N_TOP_FEATURES = 30

_extract = tldextract.TLDExtract(suffix_list_urls=())


def domain_of(url):
    return _extract(urlparse(url).netloc or url).top_domain_under_public_suffix


def load_labeled(paths):
    by_url = {}
    used = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  (skipping absent {path})")
            continue
        used.append(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    by_url.setdefault(obj["url"], obj["label"])
    urls = list(by_url)
    labels = [by_url[u] for u in urls]
    domains = [domain_of(u) for u in urls]
    return urls, labels, domains, used


def fit(X_train, y_train):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
    Xv = vec.fit_transform(X_train)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(Xv, y_train)
    return vec, clf


def legal_probs(vec, clf, X):
    idx = list(clf.classes_).index("legal")
    return clf.predict_proba(vec.transform(X))[:, idx]


def sweep(y_true_bin, probs, thresholds=THRESHOLDS):
    out = []
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        out.append({
            "threshold": t,
            "precision": precision_score(y_true_bin, preds, zero_division=0),
            "recall": recall_score(y_true_bin, preds, zero_division=0),
            "f1": f1_score(y_true_bin, preds, zero_division=0),
            "n_flagged": sum(preds),
        })
    return out


def print_sweep(title, rows):
    print(f"\n{title}")
    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'f1':>7} {'flagged':>8}")
    for r in rows:
        print(f"{r['threshold']:>7.2f} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f1']:>7.3f} {r['n_flagged']:>8}")


def report_concentration(labels, domains):
    legal = Counter(d for d, l in zip(domains, labels) if l == "legal")
    total = sum(legal.values())
    print(f"\n--- positive-class concentration ---")
    print(f"legal examples: {total} across {len(legal)} registered domains")
    cum = 0
    for i, (dom, n) in enumerate(legal.most_common(12), 1):
        cum += n
        print(f"{i:>3}. {dom:<32} {n:>5}  cum {cum / total:.1%}")
    top3 = sum(n for _, n in legal.most_common(3)) / total
    print(f"top-3 share: {top3:.1%}")
    return legal


def random_split_eval(urls, labels):
    X_tr, X_te, y_tr, y_te = train_test_split(
        urls, labels, test_size=0.2, random_state=SEED, stratify=labels)
    vec, clf = fit(X_tr, y_tr)
    probs = legal_probs(vec, clf, X_te)
    y_bin = [1 if y == "legal" else 0 for y in y_te]
    print(f"\ntest rows: {len(X_te)} ({sum(y_bin)} legal / {len(y_bin) - sum(y_bin)} non_legal)")
    return sweep(y_bin, probs)


def grouped_split_eval(urls, labels, domains, holdout_frac=0.25):
    """Hold whole registered domains out. Domains are picked so the held-out
    side carries a usable number of legal examples - a random domain draw can
    easily reserve only tail domains and leave nothing to measure recall on."""
    legal_by_dom = Counter(d for d, l in zip(domains, labels) if l == "legal")
    ranked = [d for d, _ in legal_by_dom.most_common()]
    # take every 4th domain by legal volume: spreads the holdout across head
    # and tail instead of reserving one giant domain or only tiny ones
    held = set(ranked[1::4])
    if not held:
        print("\n(not enough distinct legal domains for a grouped split)")
        return None

    tr = [i for i, d in enumerate(domains) if d not in held]
    te = [i for i, d in enumerate(domains) if d in held]
    X_tr = [urls[i] for i in tr]; y_tr = [labels[i] for i in tr]
    X_te = [urls[i] for i in te]; y_te = [labels[i] for i in te]

    if len(set(y_tr)) < 2 or "legal" not in y_te:
        print("\n(grouped split degenerate - one side lost a whole class)")
        return None

    vec, clf = fit(X_tr, y_tr)
    probs = legal_probs(vec, clf, X_te)
    y_bin = [1 if y == "legal" else 0 for y in y_te]
    print(f"\nheld-out domains ({len(held)}): {', '.join(sorted(held))}")
    print(f"test rows: {len(X_te)} ({sum(y_bin)} legal / {len(y_bin) - sum(y_bin)} non_legal)")
    return sweep(y_bin, probs)


def lodo_eval(urls, labels, domains, legal_counts):
    """Retrain without each legal-bearing domain, score only that domain."""
    targets = [d for d, n in legal_counts.items() if n >= MIN_LEGAL_FOR_LODO]
    targets.sort(key=lambda d: -legal_counts[d])

    print(f"\n--- leave-one-domain-out (threshold {OPERATING_THRESHOLD}) ---")
    print(f"{'held-out domain':<32} {'legal':>6} {'recall':>8} {'prec':>7} {'n_flag':>7}")
    results = []
    for dom in targets:
        tr = [i for i, d in enumerate(domains) if d != dom]
        te = [i for i, d in enumerate(domains) if d == dom]
        y_tr = [labels[i] for i in tr]
        if len(set(y_tr)) < 2:
            continue
        vec, clf = fit([urls[i] for i in tr], y_tr)
        probs = legal_probs(vec, clf, [urls[i] for i in te])
        y_bin = [1 if labels[i] == "legal" else 0 for i in te]
        preds = [1 if p >= OPERATING_THRESHOLD else 0 for p in probs]
        r = recall_score(y_bin, preds, zero_division=0)
        p = precision_score(y_bin, preds, zero_division=0)
        results.append((dom, legal_counts[dom], r, p, sum(preds)))
        print(f"{dom:<32} {legal_counts[dom]:>6} {r:>8.3f} {p:>7.3f} {sum(preds):>7}")

    if results:
        macro_r = sum(r for _, _, r, _, _ in results) / len(results)
        print(f"{'macro-average recall':<32} {'':>6} {macro_r:>8.3f}")
    return results


def coefficient_audit(urls, labels, domains):
    """Top positive features, flagged when they are substrings of a training
    hostname. A top-weighted feature like 'nell.' is the model naming a
    publisher, not learning what a legal URL looks like."""
    vec, clf = fit(urls, labels)
    names = vec.get_feature_names_out()
    idx = list(clf.classes_).index("legal")
    coefs = clf.coef_[0] if clf.coef_.shape[0] == 1 else clf.coef_[idx]
    # with a single coef row, positive weight points at clf.classes_[1]
    if clf.coef_.shape[0] == 1 and list(clf.classes_)[1] != "legal":
        coefs = -coefs

    hosts = {(urlparse(u).netloc or "").lower() for u in urls}
    legal_hosts = {(urlparse(u).netloc or "").lower()
                   for u, l in zip(urls, labels) if l == "legal"}

    top = sorted(zip(names, coefs), key=lambda x: -x[1])[:N_TOP_FEATURES]
    print(f"\n--- top {N_TOP_FEATURES} positive features ---")
    print(f"{'feature':<12} {'weight':>8}  in-host?")
    n_host = 0
    for name, w in top:
        in_legal_host = any(name in h for h in legal_hosts)
        in_any_host = any(name in h for h in hosts)
        tag = ""
        if in_legal_host:
            tag = "LEGAL-HOST" if not in_any_host else "host substring"
            n_host += 1
        print(f"{name!r:<12} {w:>8.3f}  {tag}")
    print(f"\n{n_host}/{N_TOP_FEATURES} top features are substrings of a legal "
          f"training hostname ({n_host / N_TOP_FEATURES:.0%})")
    return n_host


def main():
    print("--- loading ---")
    urls, labels, domains, used = load_labeled(LABELED_FILES)
    print(f"{len(urls)} labeled URLs from {len(used)} file(s) "
          f"({labels.count('legal')} legal, {labels.count('non_legal')} non_legal)")

    legal_counts = report_concentration(labels, domains)

    print("\n" + "=" * 68)
    print("1. RANDOM ROW SPLIT (what train_classifier.py reports)")
    print("=" * 68)
    random_rows = random_split_eval(urls, labels)
    print_sweep("random-split sweep", random_rows)

    print("\n" + "=" * 68)
    print("2. GROUPED SPLIT (whole registered domains held out)")
    print("=" * 68)
    grouped_rows = grouped_split_eval(urls, labels, domains)
    if grouped_rows:
        print_sweep("grouped-split sweep", grouped_rows)

        rnd = {r["threshold"]: r for r in random_rows}
        print(f"\n--- leakage gap at each threshold (random - grouped) ---")
        print(f"{'thresh':>7} {'d precision':>12} {'d recall':>10} {'d f1':>8}")
        for g in grouped_rows:
            r = rnd[g["threshold"]]
            print(f"{g['threshold']:>7.2f} {r['precision'] - g['precision']:>12.3f} "
                  f"{r['recall'] - g['recall']:>10.3f} {r['f1'] - g['f1']:>8.3f}")

    print("\n" + "=" * 68)
    print("3. LEAVE-ONE-DOMAIN-OUT")
    print("=" * 68)
    lodo_eval(urls, labels, domains, legal_counts)

    print("\n" + "=" * 68)
    print("4. COEFFICIENT AUDIT (fit on all data)")
    print("=" * 68)
    coefficient_audit(urls, labels, domains)


if __name__ == "__main__":
    main()

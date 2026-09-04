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
import argparse
import json
import os
from collections import Counter
from urllib.parse import urlparse

import tldextract
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from features import MODES, url_features

SEED = 42
OPERATING_THRESHOLD = 0.85
THRESHOLDS = [0.5, 0.65, 0.75, 0.85, 0.9]
# below this a domain's leave-one-out recall is too noisy to read
MIN_LEGAL_FOR_LODO = 5
N_TOP_FEATURES = 30

_extract = tldextract.TLDExtract(suffix_list_urls=())


def domain_of(url):
    return _extract(urlparse(url).netloc or url).top_domain_under_public_suffix


# set once by main() from --features; every fit in the run uses the same one
_MAKE_VEC = url_features


def load_labeled(mode):
    """Load one mode's rows as parallel lists.

    urls is kept alongside docs because the two diverge in text mode: the
    model sees the page body, but domain grouping and the coefficient audit
    still need to know which publisher a row came from. Rows whose extraction
    failed carry a null text and are dropped here rather than silently
    vectorising as empty strings.
    """
    _, path, field = MODES[mode]
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} - run src/text/fetch_warc_text.py first")

    by_url, dropped = {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc = obj.get(field)
            if not doc:
                dropped += 1
                continue
            by_url.setdefault(obj["url"], (doc, obj["label"]))

    urls = list(by_url)
    docs = [by_url[u][0] for u in urls]
    labels = [by_url[u][1] for u in urls]
    domains = [domain_of(u) for u in urls]
    if dropped:
        print(f"  dropped {dropped} rows with no {field}")
    return urls, docs, labels, domains, path


def fit(X_train, y_train, domains_train=None):
    """domains_train is the training fold's domains, used by the domain-purity
    filter. It must never include a held-out publisher's rows: that would let
    the thing being tested for choose the feature set."""
    vec = _MAKE_VEC()
    Xv = vec.fit_transform(X_train, domains_train)
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


def random_split_eval(docs, labels, domains):
    X_tr, X_te, y_tr, y_te, d_tr, _ = train_test_split(
        docs, labels, domains, test_size=0.2, random_state=SEED,
        stratify=labels)
    vec, clf = fit(X_tr, y_tr, d_tr)
    probs = legal_probs(vec, clf, X_te)
    y_bin = [1 if y == "legal" else 0 for y in y_te]
    print(f"\ntest rows: {len(X_te)} ({sum(y_bin)} legal / {len(y_bin) - sum(y_bin)} non_legal)")
    return sweep(y_bin, probs)


def grouped_split_eval(docs, labels, domains):
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
    X_tr = [docs[i] for i in tr]; y_tr = [labels[i] for i in tr]
    X_te = [docs[i] for i in te]; y_te = [labels[i] for i in te]
    d_tr = [domains[i] for i in tr]

    if len(set(y_tr)) < 2 or "legal" not in y_te:
        print("\n(grouped split degenerate - one side lost a whole class)")
        return None

    vec, clf = fit(X_tr, y_tr, d_tr)
    probs = legal_probs(vec, clf, X_te)
    y_bin = [1 if y == "legal" else 0 for y in y_te]
    print(f"\nheld-out domains ({len(held)}): {', '.join(sorted(held))}")
    print(f"test rows: {len(X_te)} ({sum(y_bin)} legal / {len(y_bin) - sum(y_bin)} non_legal)")
    return sweep(y_bin, probs)


def lodo_eval(docs, labels, domains, legal_counts):
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
        vec, clf = fit([docs[i] for i in tr], y_tr,
                       [domains[i] for i in tr])
        probs = legal_probs(vec, clf, [docs[i] for i in te])
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


def coefficient_audit(docs, labels, urls):
    """Top positive features, flagged when they are substrings of a training
    hostname. A top-weighted feature like 'nell.' is the model naming a
    publisher, not learning what a legal URL looks like. The same check reads
    just as well in text mode: 'cornell' or 'vermont' surfacing as a top word
    feature is publisher chrome leaking in through the page body instead."""
    vec, clf = fit(docs, labels, [domain_of(u) for u in urls])
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
    global _MAKE_VEC
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", choices=sorted(MODES), default="url",
                    help="what the model looks at: the URL string, or the "
                         "extracted page text. Everything else is identical, "
                         "so the two runs are directly comparable.")
    args = ap.parse_args()
    _MAKE_VEC = MODES[args.features][0]

    print(f"--- loading (features: {args.features}) ---")
    urls, docs, labels, domains, path = load_labeled(args.features)
    print(f"{len(urls)} labeled rows from {path} "
          f"({labels.count('legal')} legal, {labels.count('non_legal')} non_legal)")

    legal_counts = report_concentration(labels, domains)

    print("\n" + "=" * 68)
    print("1. RANDOM ROW SPLIT (what train_classifier.py reports)")
    print("=" * 68)
    random_rows = random_split_eval(docs, labels, domains)
    print_sweep("random-split sweep", random_rows)

    print("\n" + "=" * 68)
    print("2. GROUPED SPLIT (whole registered domains held out)")
    print("=" * 68)
    grouped_rows = grouped_split_eval(docs, labels, domains)
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
    lodo_eval(docs, labels, domains, legal_counts)

    print("\n" + "=" * 68)
    print("4. COEFFICIENT AUDIT (fit on all data)")
    print("=" * 68)
    coefficient_audit(docs, labels, urls)


if __name__ == "__main__":
    main()

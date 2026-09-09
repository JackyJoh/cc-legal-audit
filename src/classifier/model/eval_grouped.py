"""
Diagnostic: does the classifier recognise legal text, or just the handful of
publishers it was trained on?

The legal labels are concentrated in a few sites (cornell.edu, justice.gc.ca
and virginia.gov are most of them). A normal 80/20 split shuffles rows, so
those sites end up on both sides of it and the model can score well by
learning their quirks. That tells you nothing about a legal site it has never
seen. This script measures the difference by changing what goes in the
holdout, and nothing else.

Usage: python src/classifier/model/eval_grouped.py --features url|text

Four sections, printed in order:

1. RANDOM ROW SPLIT. 80/20 over rows, publisher ignored. Reproduces the
   headline numbers. Read it as a ceiling, not as performance.

2. GROUPED SPLIT. Same split, but the holdout is picked by publisher: choose
   a set of domains, send every row from those domains to the test side. The
   model sees none of them. Trains once, prints one threshold sweep in the
   same shape as (1), so the two subtract. That difference is the leakage.

3. LEAVE-ONE-DOMAIN-OUT. Same idea as (2), but one publisher at a time with a
   retrain for each: drop cornell.edu, train on the rest, score cornell.edu's
   rows only; repeat for the next domain. Needed because (2) pools its
   holdout into a single number that one big domain can dominate, which hides
   whether the small publishers recover. This shows each one separately.

4. COEFFICIENT AUDIT. Highest positive-weight features, fit on all data,
   flagged when the feature is a substring of a training hostname. A top
   feature like 'nell.' means the model is naming a publisher rather than
   reading legal language. In text mode the same check catches site
   boilerplate leaking in through the page body.

Everything groups by registered domain, not hostname, so law.cornell.edu and
www.law.cornell.edu count as one publisher. Holding out only one of them
would leave the other in training.

These are diagnostics. train_classifier.py reports the shipped model's
metrics.
"""
import argparse
from collections import Counter
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from features import (MODES, domain_of, domain_weights, load_labeled,
                      make_classifier, read_jsonl, url_features)

SEED = 42
OPERATING_THRESHOLD = 0.85
THRESHOLDS = [0.5, 0.65, 0.75, 0.85, 0.9]
# below this a domain's leave-one-out recall is too noisy to read
MIN_LEGAL_FOR_LODO = 5
N_TOP_FEATURES = 30

# set once by main() from --features; every fit in the run uses the same one
_MAKE_VEC = url_features
# set once by main() from --domain-weight; every fit in the run uses the same one
_DOMAIN_WEIGHT = 0.0


def fit(X_train, y_train, domains_train=None):
    """domains_train is the training fold's domains, used by the domain-purity
    filter. It must never include a held-out publisher's rows: that would let
    the thing being tested for choose the feature set."""
    vec = _MAKE_VEC()
    Xv = vec.fit_transform(X_train, domains_train)
    clf = make_classifier()
    weights = (domain_weights(y_train, domains_train, _DOMAIN_WEIGHT)
               if domains_train is not None else None)
    clf.fit(Xv, y_train, sample_weight=weights)
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
    global _MAKE_VEC, _DOMAIN_WEIGHT
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--features", choices=sorted(MODES), default="url",
                    help="what the model looks at: the URL string, or the "
                         "extracted page text. Everything else is identical, "
                         "so the two runs are directly comparable.")
    ap.add_argument("--domain-weight", type=float, default=0.5,
                    help="how much to even out publisher influence on the "
                         "fit, 0 to 1. 0 is an unweighted fit. 1 gives every "
                         "publisher the same total weight within its class, "
                         "so justice.gc.ca's 342 legal rows count no more "
                         "than a state legislature's 5. The default matches "
                         "train_classifier.py, so a run with no flag measures "
                         "the model that actually ships; a mismatch here "
                         "reports numbers for a model nobody deploys.")
    ap.add_argument("--exclude", default=None,
                    help="jsonl of URLs to drop before training, one 'url' "
                         "field per row. Running with and without a batch "
                         "measures what that batch changed, using the same "
                         "code for both numbers so the difference cannot come "
                         "from anything else.")
    args = ap.parse_args()
    _MAKE_VEC = MODES[args.features][0]
    _DOMAIN_WEIGHT = args.domain_weight

    print(f"--- loading (features: {args.features}, "
          f"domain-weight: {args.domain_weight}) ---")
    urls, docs, labels, domains, path = load_labeled(args.features)
    if args.exclude:
        drop = {r["url"] for r in read_jsonl(args.exclude)}
        keep = [i for i, u in enumerate(urls) if u not in drop]
        n_before = len(urls)
        urls = [urls[i] for i in keep]
        docs = [docs[i] for i in keep]
        labels = [labels[i] for i in keep]
        domains = [domains[i] for i in keep]
        print(f"excluding {args.exclude}: dropped {n_before - len(urls)} rows")
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

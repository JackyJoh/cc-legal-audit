"""
Reports the tail of the TF-IDF score distribution over the pages Laya keeps,
so T can be set by hand.

Usage:
  python src/classifier/typesafe/screen_threshold.py
  python src/classifier/typesafe/screen_threshold.py --cache X --output Y

Scores the open-web text cache with the frozen TF-IDF bundle and prints three
tables:

  1. The lowest scores among the protected set, with the gap below each. The
     protected set is every page Laya keeps at its frozen cut, since a page
     the judge would reject anyway costs nothing to screen out.
  2. The forwarded rate at each candidate T, as a multiple of the rate at the
     observed minimum. This is what a margin costs in Laya time.
  3. Protected pages lost at each T.

The candidates are the observed minimum, the minimum less one to five mean
gaps, and a fixed grid. The minimum is an order statistic, not a floor: at
n=172 it estimates the 0.6% quantile, so T is set below it by a margin read
off the gaps, which run inversely to the tail's density.

Nothing is fit, tuned, or written. T is chosen by hand and frozen.
"""
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
from features import legal_probs, load_bundle, read_jsonl  # noqa: E402

DEFAULT_CACHE = "data/candidates/_pool_text_cache.jsonl"
DEFAULT_LAYA = "data/candidates/laya_unseen_scores.jsonl"
DEFAULT_MODEL = "models/text_clf.joblib"
HAND_LABELS = ("data/candidates/jev_openweb_kept_full.jsonl",
               "data/candidates/jev_openweb_060_090_full.jsonl")

# 0.37 through 0.50 line the grid up with the superseded screener table in
# the classifier README.
T_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.37, 0.40, 0.45, 0.50]
# Mean gap is also read at each of these depths: too few gaps make the mean
# noisy, too many measure the body instead of the tail.
TAIL_DEPTHS = (5, 10, 20)
MARGINS = (1, 2, 3, 4, 5)


def load_cache(path):
    """Every cached page carrying text, deduped by URL.

    compact() leaves retried duplicates in the cache with the last one live,
    so a dict keyed on URL in file order is the reader the writer assumes.
    Pages without text stay out: they cannot be scored, and counting them in
    the denominator would understate the forwarded rate.
    """
    rows = {}
    for r in read_jsonl(path):
        rows[r["url"]] = r
    kept = {u: r for u, r in rows.items() if r.get("text")}
    return kept, len(rows), len(rows) - len(kept)


def load_hand_labels():
    labels = {}
    for path in HAND_LABELS:
        if not os.path.exists(path):
            print(f"  note: {path} missing, cannot verify the protected set")
            continue
        for r in read_jsonl(path):
            labels[r["url"]] = r["label"]
    return labels


def check_extractor(bundle, rows):
    """Refuse to score text produced by an extractor the model never saw."""
    trained_on = set(bundle["meta"].get("extractors") or [])
    seen = {r.get("extractor") for r in rows.values() if r.get("extractor")}
    if not trained_on or not seen:
        print("  note: extractor unverifiable (missing on the model or the cache)")
        return
    unknown = seen - trained_on
    if unknown:
        raise SystemExit(f"extractor mismatch: cache has {sorted(unknown)}, "
                         f"model trained on {sorted(trained_on)}")


def mean_gap(low):
    return statistics.mean(b - a for a, b in zip(low, low[1:]))


def tail_report(scores, depth):
    """The lowest `depth` scores, the gaps between them, and what they imply."""
    ordered = sorted(scores)
    low = ordered[:depth]
    gaps = [b - a for a, b in zip(low, low[1:])]

    print(f"\n--- positive lower tail (n={len(scores)}, lowest {len(low)}) ---")
    print(f"{'rank':>5} {'score':>9} {'gap below':>11}")
    for i, s in enumerate(low, 1):
        gap = f"{s - low[i - 2]:.4f}" if i > 1 else "-"
        print(f"{i:>5} {s:>9.4f} {gap:>11}")

    mg, med = mean_gap(low), statistics.median(gaps)
    print(f"\nmean gap {mg:.4f}   median gap {med:.4f}   bottom gap {gaps[0]:.4f}")

    stability = "   ".join(
        f"k={k}: {mean_gap(ordered[:k]):.4f}"
        for k in TAIL_DEPTHS if len(ordered) > k)
    print(f"mean gap by depth   {stability}")
    print("  if these disagree, prefer the shallower reading")

    print(f"observed min {low[0]:.4f} estimates the "
          f"{1 / (len(scores) + 1):.2%} quantile of legal-page scores")
    if gaps[0] > 2 * med:
        print("  bottom gap over twice the median: the minimum is a straggler "
              "out of a sparse region, so the tail runs well below it and the "
              "margin should be large")
    else:
        print("  bottom gap in line with the rest: dense tail, the minimum is "
              "comparatively well pinned")
    return low[0], mg


def cost_table(probs, candidates, floor):
    """Forwarded rate at each candidate T, priced against the observed min."""
    n = len(probs)
    base = sum(1 for p in probs if p >= floor) / n
    print(f"\n--- forwarded rate over {n:,} crawl pages ---")
    print(f"{'T':>7} {'forwarded':>10} {'rate':>9} {'vs min':>8}  note")
    for t, note in candidates:
        rate = sum(1 for p in probs if p >= t) / n
        mult = f"{rate / base:.2f}x" if base else "-"
        print(f"{t:>7.4f} {int(round(rate * n)):>10,} {rate:>8.3%} "
              f"{mult:>8}  {note}")

    print(f"\n'vs min' is Laya time relative to T = {floor:.4f}. A margin is "
          f"cheap only where this column stays flat.")


def loss_table(pos, candidates):
    print("\n--- losses on the protected set ---")
    print(f"{'T':>7} {'lost':>6}  of {len(pos)}")
    for t, _ in candidates:
        lost = sum(1 for p in pos if p < t)
        if lost or abs(t - min(pos)) < 1e-9:
            print(f"{t:>7.4f} {lost:>6}")
    print("\nZero at the observed minimum is true by construction, not "
          "evidence. The margin is what covers pages this sample never held.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--cache", default=DEFAULT_CACHE,
                    help="open-web text cache written by openweb_precision.py")
    ap.add_argument("--laya", default=DEFAULT_LAYA,
                    help="Laya scores over the unseen open-web pages")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="the frozen TF-IDF bundle")
    ap.add_argument("--laya-cut", type=float, default=0.85,
                    help="Laya's frozen cut, which defines the protected set")
    ap.add_argument("--tail", type=int, default=10,
                    help="how many of the lowest positives to read gaps from")
    ap.add_argument("--output", help="jsonl of every scored page")
    args = ap.parse_args()

    bundle = load_bundle(args.model)

    rows, n_cached, n_no_text = load_cache(args.cache)
    print(f"\ncache: {n_cached:,} pages, {n_no_text:,} without text, "
          f"{len(rows):,} scorable")
    if not rows:
        raise SystemExit(f"no page in {args.cache} carries text")
    check_extractor(bundle, rows)

    protected = {r["url"] for r in read_jsonl(args.laya)
                 if r["p_legal"] >= args.laya_cut}
    print(f"protected set: {len(protected)} pages Laya keeps at "
          f"p >= {args.laya_cut}")

    labels = load_hand_labels()
    if labels:
        unverified = [u for u in protected if labels.get(u) != "legal"]
        if unverified:
            print(f"  WARNING: {len(unverified)} of them are not hand-labeled "
                  f"legal; the protected set is meant to be fully verified")

    missing = sorted(u for u in protected if u not in rows)
    if missing:
        print(f"  WARNING: {len(missing)} protected pages have no text in the "
              f"cache and cannot be measured:")
        for u in missing[:10]:
            print(f"    {u}")

    urls = list(rows)
    probs = [float(p) for p in
             legal_probs(bundle, [rows[u]["text"] for u in urls])]
    by_url = dict(zip(urls, probs))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for u in urls:
                f.write(json.dumps({
                    "url": u,
                    "prob_legal": round(by_url[u], 6),
                    "protected": u in protected,
                    "text_chars": rows[u].get("text_chars"),
                }) + "\n")
        print(f"wrote {args.output}")

    pos = [by_url[u] for u in protected if u in by_url]
    if len(pos) < TAIL_DEPTHS[0]:
        raise SystemExit(f"only {len(pos)} protected pages scored, too few to "
                         f"read a tail")

    floor, mg = tail_report(pos, args.tail)

    candidates = {round(floor, 4): f"observed min, loses 0 of {len(pos)}"}
    for m in MARGINS:
        candidates.setdefault(round(floor - m * mg, 4),
                              f"min - {m} mean gap{'s' if m > 1 else ''}")
    for t in T_GRID:
        candidates.setdefault(round(t, 4), "")
    candidates = sorted(candidates.items(), reverse=True)

    cost_table(probs, candidates, floor)
    loss_table(pos, candidates)


if __name__ == "__main__":
    main()

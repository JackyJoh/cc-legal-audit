"""
Turns the hand-labeled legal-domain sample into the deployment numbers: what
fraction of the pages the classifier keeps are really legal documents, and what
it throws away to get there.

The sample was drawn as three score strata at different rates, so a raw count
over the labels would misread the pool: the low stratum holds most of the pool
but only 50 labels, and the high stratum the reverse. Every labeled page
therefore carries a weight, the number of pool pages one label stands for,
which is that stratum's frame size over the number of labels drawn from it.
Precision at a threshold is the weighted legal count over the weighted kept
count, which is unbiased at any threshold, not only at a stratum boundary.

Weights use each stratum's available count rather than its raw pool count. The
difference is the pages dropped for having been labeled in an earlier batch,
and those were selected by the old model rather than at random, so the frame
actually sampled from is the one that supports an unbiased estimate.

Intervals come from a stratified bootstrap, resampling labels within each
stratum, since a closed form for a ratio of weighted sums across strata is
not worth deriving for 250 rows. Precision and recall both get one, and the
recall interval is much the wider: precision is decided by the pages the
filter keeps, which are concentrated in the two heavily-labeled strata,
while recall is measured against the estimated legal count for the whole
frame, and about a quarter of that rests on the few legal pages found in the
sparsely-labeled low stratum.

Reads:
  data/candidates/legal_sample_batch.jsonl   URLs with hand labels
  data/candidates/legal_sample_scores.jsonl  score and stratum per URL
  data/candidates/legal_sample_stats.json    per-stratum frame sizes
"""
import json
import math
import random
from collections import defaultdict

BATCH_FILE   = "data/candidates/legal_sample_batch.jsonl"
SCORES_FILE  = "data/candidates/legal_sample_scores.jsonl"
STATS_FILE   = "data/candidates/legal_sample_stats.json"
COUNTS_FILE  = "data/candidates/cc_domain_counts.jsonl"
DOMAINS_FILE = "data/candidates/legal_domains.jsonl"

THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CHOSEN = 0.75
N_BOOT = 4000
SEED = 42


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(k, n, z=1.96):
    """Binomial interval that stays inside 0 to 1 at small n, unlike normal."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def rates(rows, t):
    """Weighted kept count, weighted legal-and-kept count, weighted legal total."""
    kept = legal_kept = legal_all = 0.0
    for r in rows:
        legal = r["label"] == "legal"
        legal_all += r["w"] * legal
        if r["prob"] >= t:
            kept += r["w"]
            legal_kept += r["w"] * legal
    return kept, legal_kept, legal_all


def boot_cis(by_stratum, t):
    """Percentile intervals for precision and recall, resampling inside each
    stratum.

    Both come off the same draws, so the two intervals describe one bootstrap
    rather than two independent ones, and the run costs a single pass.

    Recall is the wider of the two by a lot, and structurally so: its
    denominator is the estimated legal count for the whole frame, and roughly
    a quarter of that estimate rests on the handful of legal pages found in
    the low stratum, each standing for ~96 pool pages. Precision at a high
    threshold barely touches those rows; recall cannot avoid them.
    """
    rng = random.Random(SEED)
    prec, rec = [], []
    for _ in range(N_BOOT):
        draw = []
        for rows in by_stratum.values():
            draw.extend(rng.choices(rows, k=len(rows)))
        kept, legal_kept, legal_all = rates(draw, t)
        prec.append(legal_kept / kept if kept else 0.0)
        rec.append(legal_kept / legal_all if legal_all else 0.0)
    prec.sort()
    rec.sort()
    lo, hi = int(0.025 * N_BOOT), int(0.975 * N_BOOT)
    return (prec[lo], prec[hi]), (rec[lo], rec[hi])


def main():
    labels = {r["url"]: r["label"] for r in load_jsonl(BATCH_FILE)}
    scores = {r["url"]: r for r in load_jsonl(SCORES_FILE)}
    stats = json.load(open(STATS_FILE, encoding="utf-8"))

    unlabeled = [u for u, v in labels.items() if v not in ("legal", "non_legal")]
    if unlabeled:
        raise SystemExit(f"{len(unlabeled)} rows not labeled, first: {unlabeled[0]}")

    by_stratum = defaultdict(list)
    for url, label in labels.items():
        s = scores[url]
        by_stratum[s["stratum"]].append(
            {"url": url, "prob": s["prob_legal"], "label": label, "stratum": s["stratum"]})

    print("=" * 68)
    print("1. WHAT WAS LABELED, BY STRATUM")
    print("=" * 68)
    print(f"{'stratum':>8} {'range':>12} {'frame':>7} {'labels':>7} {'legal':>6} "
          f"{'rate':>7}  95% CI")
    for name in ("high", "mid", "low"):
        rows = by_stratum[name]
        st = stats["strata"][name]
        frame = st["available"]
        k = sum(1 for r in rows if r["label"] == "legal")
        lo, hi = wilson(k, len(rows))
        w = frame / len(rows)
        for r in rows:
            r["w"] = w
        band = f"{st['low']:.2f}-{st['high']:.2f}"
        print(f"{name:>8} {band:>12} "
              f"{frame:>7} {len(rows):>7} {k:>6} {k/len(rows):>7.3f}  "
              f"[{lo:.3f}, {hi:.3f}]")

    allrows = [r for rows in by_stratum.values() for r in rows]
    frame_total = sum(stats["strata"][n]["available"] for n in by_stratum)
    _, _, legal_all = rates(allrows, 0.0)
    print(f"\nframe total: {frame_total} pages")
    print(f"estimated legal pages in frame: {legal_all:,.0f} "
          f"({legal_all / frame_total:.1%} base rate)")

    print("\n" + "=" * 68)
    print("2. PRECISION, RECALL AND YIELD BY THRESHOLD")
    print("=" * 68)
    print(f"{'t':>5} {'precision':>10} {'95% CI':>16} {'contam':>7} "
          f"{'recall':>7} {'95% CI':>16} {'yield':>7} {'kept':>8}")
    for t in THRESHOLDS:
        kept, legal_kept, _ = rates(allrows, t)
        if kept == 0:
            continue
        prec = legal_kept / kept
        rec = legal_kept / legal_all
        (plo, phi), (rlo, rhi) = boot_cis(by_stratum, t)
        print(f"{t:>5.2f} {prec:>10.3f} {f'[{plo:.3f}, {phi:.3f}]':>16} "
              f"{1 - prec:>7.1%} {rec:>7.3f} {f'[{rlo:.3f}, {rhi:.3f}]':>16} "
              f"{kept / frame_total:>7.1%} {kept:>8,.0f}")

    print("\n" + "=" * 68)
    print("3. WHAT A HIGH THRESHOLD DISCARDS")
    print("=" * 68)
    missed = sorted([r for r in allrows if r["label"] == "legal" and r["prob"] < 0.85],
                    key=lambda r: -r["prob"])
    print(f"labeled legal but scoring under 0.85: {len(missed)} of "
          f"{sum(1 for r in allrows if r['label'] == 'legal')} labeled legal")
    host = defaultdict(int)
    for r in missed:
        host[r["url"].split("/")[2].replace("www.", "")] += 1
    print("\nby host (a single host dominating means the loss is systematic, "
          "not spread):")
    for h, n in sorted(host.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {n:>3}  {h}")
    print("\nlowest-scoring legal pages, to eyeball whether they are one "
          "document type:")
    for r in missed[-12:]:
        print(f"  {r['prob']:.3f}  {r['url'][:100]}")

    at_scale(allrows, legal_all, frame_total, by_stratum)


def at_scale(allrows, legal_all, frame_total, by_stratum):
    """Projects the sample's rates onto every eligible page in the crawl.

    The sample is a uniform draw from the same domains, so its yield and
    precision carry over to the full set of eligible pages: yield times
    eligible is how many pages the corpus keeps, and precision decides how
    many of those are really legal documents. The false-positive rate is
    reported against the non-legal pages rather than against everything,
    since that is the share of junk the filter lets through, and it looks far
    smaller than contamination because non-legal pages outnumber legal ones
    in the pool.
    """
    counts = load_jsonl(COUNTS_FILE)
    kept, legal_kept, _ = rates(allrows, CHOSEN)
    yield_rate = kept / frame_total
    precision = legal_kept / kept
    base_rate = legal_all / frame_total
    (plo, phi), _ = boot_cis(by_stratum, CHOSEN)

    print("\n" + "=" * 68)
    print(f"4. AT SCALE, AT THRESHOLD {CHOSEN}")
    print("=" * 68)
    print(f"{'type':>12} {'domains':>8} {'eligible':>10} {'kept':>9} "
          f"{'legal docs':>11} {'false pos':>10}")
    total = defaultdict(float)
    for t in ("court", "legislature"):
        elig = sum(c["n_eligible"] for c in counts if c["type"] == t)
        n_dom = sum(1 for c in counts if c["type"] == t and c["n_eligible"] > 0)
        k = elig * yield_rate
        print(f"{t:>12} {n_dom:>8} {elig:>10,} {k:>9,.0f} "
              f"{k * precision:>11,.0f} {k * (1 - precision):>10,.0f}")
        total["eligible"] += elig
        total["domains"] += n_dom
    elig = total["eligible"]
    kept_all = elig * yield_rate
    print(f"{'total':>12} {int(total['domains']):>8} {elig:>10,.0f} "
          f"{kept_all:>9,.0f} {kept_all * precision:>11,.0f} "
          f"{kept_all * (1 - precision):>10,.0f}")

    non_legal = elig * (1 - base_rate)
    fpr = kept_all * (1 - precision) / non_legal
    print(f"\nbase rate in these domains  : {base_rate:.1%}")
    print(f"contamination of the corpus : {1 - precision:.1%} "
          f"(range {1 - phi:.1%} to {1 - plo:.1%})")
    print(f"false-positive rate         : {fpr:.2%} of the "
          f"{non_legal:,.0f} non-legal pages")
    print(f"expected legal documents    : {kept_all * precision:,.0f} "
          f"(range {kept_all * plo:,.0f} to {kept_all * phi:,.0f})")

    # Per-domain projections say whether the corpus is spread over publishers
    # or concentrated in a few, which is what the strata check depends on.
    per_dom = sorted((c["n_eligible"] * yield_rate for c in counts), reverse=True)
    print(f"\ndomains by projected document count:")
    for floor in (5000, 1000, 500, 200, 100, 50, 10):
        print(f"  >= {floor:>5} docs: {sum(1 for d in per_dom if d >= floor):>4}")
    top10 = sum(per_dom[:10]) / sum(per_dom)
    print(f"top 10 domains hold {top10:.0%} of the corpus")


if __name__ == "__main__":
    main()

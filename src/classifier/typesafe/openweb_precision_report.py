"""
Reports Jev's open-web precision at score-threshold intervals, using only
the two files that carry real hand labels:

  data/candidates/jev_openweb_kept_full.jsonl     p_legal >= 0.90
  data/candidates/jev_openweb_060_090_full.jsonl  0.60 <= p_legal < 0.90

Nothing else counts as truth here - not run_*.jsonl, not labeled_urls.jsonl,
not legal_sample_batch.jsonl or deployment_sample*.jsonl. Every label in
those traces through an LLM somewhere, and validating one LLM against
another LLM's output isn't a precision measurement. A row still carrying
the unfilled placeholder ("your_label", or null from an older run) is
skipped, not counted as either class - the file may still be mid-label.

For each threshold t, precision is legal-count over labeled-count among
rows with p_legal >= t, i.e. "if the cascade's Jev accept-threshold were
t, what precision would the admitted set have". This is cumulative from
the top, same convention as src/classifier/validation/precision_report.py,
so t=0.90 here should reproduce that file's kept_full result exactly
(160/160) once both bands are fully labeled.

Usage:
  python src/classifier/typesafe/openweb_precision_report.py
"""
import json
import math

SCORES = "data/labels/jev_openweb_scores.jsonl"
HAND_LABEL_FILES = [
    "data/candidates/jev_openweb_kept_full.jsonl",
    "data/candidates/jev_openweb_060_090_full.jsonl",
]
PLACEHOLDERS = {None, "your_label"}

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(k, n, z=1.96):
    """Lower/upper 95% bound on a proportion; matches precision_report.py."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - half), min(1.0, c + half)


def main():
    p_by_url = {}
    for r in load_jsonl(SCORES):
        p_by_url[r["url"]] = r["p_legal"]   # last line wins, project convention

    truth = {}
    skipped = 0
    for path in HAND_LABEL_FILES:
        for r in load_jsonl(path):
            if r["label"] in PLACEHOLDERS:
                skipped += 1
                continue
            if r["label"] not in ("legal", "non_legal"):
                raise SystemExit(f"unexpected label {r['label']!r} in {path} for {r['url']}")
            truth[r["url"]] = r["label"] == "legal"

    missing_score = [u for u in truth if u not in p_by_url]
    if missing_score:
        raise SystemExit(f"{len(missing_score)} hand-labeled urls have no p_legal in {SCORES}, "
                          f"e.g. {missing_score[0]}")

    rows = [(p_by_url[u], t) for u, t in truth.items()]
    print(f"hand-labeled: {len(rows)}   still placeholder (skipped): {skipped}")
    print(f"  {sum(t for _, t in rows)} legal, {sum(not t for _, t in rows)} non_legal")
    print()
    print(f"{'t':>6}{'n':>7}{'legal':>8}{'precision':>11}{'wilson 95% CI':>22}")
    print("-" * 54)
    for t in THRESHOLDS:
        band = [lab for p, lab in rows if p >= t]
        n = len(band)
        k = sum(band)
        if n == 0:
            print(f"{t:>6.2f}{n:>7}{'-':>8}{'-':>11}{'-':>22}")
            continue
        lo, hi = wilson(k, n)
        print(f"{t:>6.2f}{n:>7}{k:>8}{k/n:>11.4f}   [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()

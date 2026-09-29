"""
Scores Jev's is_legal answers against a file of labels held as truth.

Joins the predictions to the truth on url. Jev's call is the direction
p_legal leans (>= 0.5 legal, else non_legal); its margin is how far p sits
from the pole it picked (min(p, 1-p): 0.01 means p was 0.01 or 0.99). The
report is accuracy by margin band, so it answers "when Jev is this sure, how
often is it right?" and, cumulatively, "if I only trusted calls within this
margin, what would I keep and how clean would it be?".

Two truth formats are accepted, since the project has both:
  - a `label` field holding "legal" / "non_legal"
        data/processed/labeled_text.jsonl, data/labels/run_*.jsonl
  - a `legal` or `non_legal` key whose presence is the label (value is a note)
        data/validation/deployment_sample*.jsonl (hand-labeled rounds)

Usage:
    python src/classifier/typesafe/eval_is_legal.py
    python src/classifier/typesafe/eval_is_legal.py --pred data/labels/other.jsonl
    python src/classifier/typesafe/eval_is_legal.py --truth data/validation/deployment_sample.jsonl
    python src/classifier/typesafe/eval_is_legal.py --bands 0.01 0.02 0.03 0.04 0.1 0.5
    python src/classifier/typesafe/eval_is_legal.py --show-errors

Output is console only.
"""
import argparse
import json

DEFAULT_PRED  = "data/labels/jev_is_legal.jsonl"
DEFAULT_TRUTH = "data/processed/labeled_text.jsonl"

# Upper edges of the margin bands. The fine 0.01 steps are where Jev parks
# its confident answers; the coarser bands cover the rest up to the 0.5
# midpoint so every row lands somewhere.
DEFAULT_BANDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5]


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def truth_label(row):
    """True for legal, False for non_legal, None if the row carries no label."""
    if "label" in row:
        return {"legal": True, "non_legal": False}.get(row["label"])
    if "legal" in row:
        return True
    if "non_legal" in row:
        return False
    return None


def load_truth(path):
    truth, unlabeled = {}, 0
    for row in iter_jsonl(path):
        lab = truth_label(row)
        if lab is None:
            unlabeled += 1
            continue
        truth[row["url"]] = lab
    return truth, unlabeled


def load_pred(path):
    return {row["url"]: row["p_legal"] for row in iter_jsonl(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pred",  default=DEFAULT_PRED,  help="labeler output jsonl")
    ap.add_argument("--truth", default=DEFAULT_TRUTH, help="jsonl of labels held as truth")
    ap.add_argument("--bands", type=float, nargs="+", default=DEFAULT_BANDS,
                    help="upper edges of margin bands (margin = min(p, 1-p))")
    ap.add_argument("--show-errors", action="store_true",
                    help="list every wrong call, tightest margin first")
    args = ap.parse_args()

    truth, unlabeled = load_truth(args.truth)
    pred = load_pred(args.pred)

    # (url, truth, p, called_legal, margin, correct)
    rows = []
    for u, p in pred.items():
        if u not in truth:
            continue
        called = p >= 0.5
        rows.append((u, truth[u], p, called, min(p, 1 - p), called == truth[u]))

    n_legal = sum(t for _, t, *_ in rows)
    print()
    print(f"  truth  {args.truth}")
    print(f"  pred   {args.pred}")
    print(f"  joined {len(rows):,} urls   legal {n_legal:,}   non_legal {len(rows) - n_legal:,}"
          f"   (pred-only {sum(u not in truth for u in pred)}, "
          f"truth-only {sum(u not in pred for u in truth)}, "
          f"unlabeled {unlabeled})")
    if not rows:
        return

    n_right = sum(r[5] for r in rows)
    print(f"  overall  {n_right:,}/{len(rows):,} right  acc {n_right / len(rows):.3f}")

    # Band table. Each row is the docs whose margin falls in [lo, hi); the
    # cumulative columns are everything with margin < hi, i.e. what you'd
    # keep if you only trusted calls at least that confident.
    edges = [0.0] + sorted(args.bands)
    print()
    print(f"  margin = distance from the pole Jev picked (p<0.5 -> non_legal, p>=0.5 -> legal)")
    print()
    print(f"  {'margin':<14}{'n':>6} {'right':>6} {'acc':>7}   {'wrong: legal':>13} {'non_legal':>10}"
          f"   |{'cum n':>7} {'cum acc':>8} {'kept':>6}")
    print("  " + "-" * 88)
    cum = []
    for lo, hi in zip(edges, edges[1:]):
        # last band is closed on the right so margin == 0.5 (p exactly 0.5)
        # isn't dropped
        last = hi == edges[-1]
        band = [r for r in rows if lo <= r[4] < hi or (last and r[4] == hi)]
        cum += band
        n = len(band)
        right = sum(r[5] for r in band)
        wrong_legal = sum(1 for r in band if not r[5] and r[1])       # legal, called no
        wrong_non   = sum(1 for r in band if not r[5] and not r[1])   # non_legal, called yes
        acc = f"{right / n:.3f}" if n else "   -"
        cum_right = sum(r[5] for r in cum)
        cum_acc = f"{cum_right / len(cum):.3f}" if cum else "   -"
        bar = "#" * round(40 * n / len(rows))
        print(f"  [{lo:.2f},{hi:.2f}) {n:>6} {right:>6} {acc:>7}   {wrong_legal:>13} {wrong_non:>10}"
              f"   |{len(cum):>7} {cum_acc:>8} {100 * len(cum) / len(rows):>5.0f}%  {bar}")
    print()
    print("  'wrong: legal' = truly legal, Jev said non_legal; 'non_legal' = the reverse")
    print()

    if args.show_errors:
        print("  wrong calls, most confident first:")
        for u, t, p, called, m, ok in sorted(rows, key=lambda r: r[4]):
            if ok:
                continue
            kind = "legal     -> said no " if t else "non_legal -> said yes"
            print(f"    {kind}  p={p:.3f}  {u}")
        print()


if __name__ == "__main__":
    main()

"""
Open-web test for the fine-tuned Laya, on the pages it never trained on.

The 223 hand labels only cover pages Jev scored >= 0.60, so they cannot show
junk Laya keeps that Jev rejected. This covers that. Jev already scored every
page of the 97k open-web draw, so once Laya scores the unseen part, each page
lands in one of four buckets:

                    Jev >= J              Jev < J
  Laya >= L         both keep             Laya-only keeps   <- hand-label these
  Laya <  L         Laya misses           both reject

Every Laya keep that is not already hand-labeled goes to a labeling file, so
open-web precision can be read at any cut from L up.

Two modes:

  --unseen      Writes the open-web urls Laya never trained on (the 223 test
                pages included, since training excluded them). The training
                selection is reproduced by calling finetune_laya.collect with
                the same --neg-sample, so the two cannot drift apart.

  --spot-check  Reads Laya's scores on those pages, prints the bucket table,
                appends unlabeled Laya keeps to the labeling file with the
                "your_label" placeholder (existing labels are never touched),
                and, once labels exist, prints Laya's open-web precision.

Runs in the laya environment:
  python src/classifier/typesafe/build_laya_eval.py --unseen
  python src/classifier/typesafe/score_laya.py --model models/laya-legal --max-len 1024 --batch-size 32 ^
      --input data/candidates/laya_unseen_openweb.jsonl data/candidates/jev_openweb_kept_full.jsonl ^
              data/candidates/jev_openweb_060_090_full.jsonl ^
      --output data/candidates/laya_unseen_scores.jsonl
  python src/classifier/typesafe/build_laya_eval.py --spot-check
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from score_laya import DEFAULT_INPUTS as HAND_FILES, iter_jsonl, wilson  # noqa: E402

UNSEEN = "data/candidates/laya_unseen_openweb.jsonl"
LAYA_SCORES = "data/candidates/laya_unseen_scores.jsonl"
JEV_SCORES = "data/labels/jev_openweb_scores.jsonl"
SPOT_FULL = "data/candidates/laya_spot_check_full.jsonl"
SPOT_SCORES = "data/candidates/laya_spot_check_scores.jsonl"
MODEL = "models/laya-legal"
PLACEHOLDERS = {None, "your_label"}
THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.99]


def write_unseen(neg_sample):
    from finetune_laya import collect
    trained = {u for u, *_ in collect(neg_sample)}
    jev = {r["url"] for r in iter_jsonl(JEV_SCORES)}
    unseen = sorted(jev - trained)
    with open(UNSEEN, "w", encoding="utf-8") as f:
        for u in unseen:
            f.write(json.dumps({"url": u}) + "\n")
    print(f"\nopen-web pages {len(jev):,}, trained on {len(jev & trained):,}, unseen {len(unseen):,}")
    print(f"wrote {UNSEEN}")


def hand_labels():
    labels = {}
    for path in HAND_FILES + [SPOT_FULL]:
        if os.path.exists(path):
            for r in iter_jsonl(path):
                if r.get("label") not in PLACEHOLDERS:
                    labels[r["url"]] = r["label"] == "legal"
    return labels


def spot_check(laya_cut, jev_cut, max_len):
    model = os.path.normpath(MODEL)
    laya = {r["url"]: r["p_legal"] for r in iter_jsonl(LAYA_SCORES)
            if r.get("max_len") == max_len and os.path.normpath(r.get("model", "")) == model}
    unseen = {r["url"] for r in iter_jsonl(UNSEEN)}
    jev = {r["url"]: r["p_legal"] for r in iter_jsonl(JEV_SCORES)}
    urls = [u for u in unseen if u in laya]
    print(f"unseen pages {len(unseen):,}, scored by Laya {len(urls):,}"
          + (f"  ({len(unseen) - len(urls):,} not scored, no text or not run yet)" if len(urls) < len(unseen) else ""))

    labels = hand_labels()
    both = [u for u in urls if laya[u] >= laya_cut and jev[u] >= jev_cut]
    laya_only = [u for u in urls if laya[u] >= laya_cut and jev[u] < jev_cut]
    misses = [u for u in urls if laya[u] < laya_cut and jev[u] >= jev_cut]
    neither = len(urls) - len(both) - len(laya_only) - len(misses)
    print(f"\n{'':<16}{'Jev >= ' + str(jev_cut):>14}{'Jev < ' + str(jev_cut):>14}")
    print(f"{'Laya >= ' + str(laya_cut):<16}{len(both):>14,}{len(laya_only):>14,}   <- Laya-only keeps")
    print(f"{'Laya < ' + str(laya_cut):<16}{len(misses):>14,}{neither:>14,}")
    if misses:
        print(f"\nLaya misses (Jev >= {jev_cut}, Laya < {laya_cut}):")
        for u in sorted(misses, key=lambda u: laya[u]):
            lab = labels.get(u)
            tag = "legal" if lab else "non_legal" if lab is False else "unlabeled"
            print(f"  laya {laya[u]:.3f}  jev {jev[u]:.3f}  {tag:<9}  {u[:100]}")

    # labeling file: every Laya keep without a hand label, appended, never rewritten
    keeps = [u for u in urls if laya[u] >= laya_cut]
    existing, rows = set(), []
    if os.path.exists(SPOT_FULL):
        for r in iter_jsonl(SPOT_FULL):
            if r["url"] not in existing:
                existing.add(r["url"])
                rows.append(r)
    new = sorted(u for u in keeps if u not in labels and u not in existing)
    rows += [{"url": u, "label": "your_label"} for u in new]
    with open(SPOT_FULL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(SPOT_SCORES, "w", encoding="utf-8") as f:
        for u in sorted(keeps):
            f.write(json.dumps({"url": u, "p_laya": laya[u], "p_jev": jev[u]}) + "\n")
    todo = sum(1 for r in rows if r.get("label") in PLACEHOLDERS)
    print(f"\nLaya keeps (>= {laya_cut}): {len(keeps):,}, already hand-labeled {sum(u in labels for u in keeps):,}")
    print(f"{SPOT_FULL}: {len(new)} added, {todo} still to label (scores held in {SPOT_SCORES})")

    print(f"\nLaya open-web precision on unseen pages (labeled keeps only)")
    print(f"{'t':>6}{'kept':>7}{'labeled':>9}{'legal':>7}{'precision':>11}{'wilson 95% CI':>22}")
    print("-" * 62)
    for t in THRESHOLDS:
        kept = [u for u in urls if laya[u] >= t]
        lab = [labels[u] for u in kept if u in labels]
        n, k = len(lab), sum(lab)
        if n == 0:
            print(f"{t:>6.2f}{len(kept):>7}{0:>9}{'-':>7}{'-':>11}{'-':>22}")
            continue
        lo, hi = wilson(k, n)
        print(f"{t:>6.2f}{len(kept):>7}{n:>9}{k:>7}{k / n:>11.4f}   [{lo:.4f}, {hi:.4f}]")
    if todo:
        print(f"\n{todo} keeps still unlabeled; precision is final once kept == labeled.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--unseen", action="store_true", help="write the open-web urls Laya never trained on")
    mode.add_argument("--spot-check", action="store_true", help="bucket table + labeling file of Laya keeps")
    ap.add_argument("--neg-sample", type=int, default=20_000, help="must match the training run")
    ap.add_argument("--laya-cut", type=float, default=0.80)
    ap.add_argument("--jev-cut", type=float, default=0.90)
    ap.add_argument("--max-len", type=int, default=1024, help="must match the scoring run")
    args = ap.parse_args()
    if args.unseen:
        write_unseen(args.neg_sample)
    else:
        spot_check(args.laya_cut, args.jev_cut, args.max_len)


if __name__ == "__main__":
    main()

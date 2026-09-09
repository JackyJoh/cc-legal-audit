"""
Scores documents with a saved model and writes one probability per row.

Usage:
  python src/classifier/score.py --model models/text_clf.joblib \
      --input data/processed/deployment_text.jsonl \
      --output data/processed/deployment_scores.jsonl

This is what turns the classifier from something that reports metrics on its
own labels into something that can be pointed at the crawl. The input is any
jsonl carrying the field the model reads ("text" for a text model, "url" for
a URL model); rows missing it are skipped and counted, never scored as empty.

The headline output is the flag rate: what fraction of the input clears the
threshold. On a uniform crawl sample that number alone bounds precision
before anything is hand-labeled. If legal pages are 0.1% of the crawl and the
model flags 3% of it, then at most one flagged page in thirty can be right,
whatever the held-out metrics said. Pass --base-rate to print that ceiling.

Extraction has to match. A text model trained on trafilatura 2.2.0 output and
pointed at text from a different extractor is not being tested on the same
kind of input it learned from, so a mismatch stops the run rather than
producing numbers that look fine.
"""
import argparse
import json
import os

from features import MODES, legal_probs, load_bundle, read_jsonl

REPORT_THRESHOLDS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95]


def check_extractors(bundle, rows, allow_mismatch):
    """Refuse to score text produced by an extractor the model never saw."""
    trained_on = set(bundle["meta"].get("extractors") or [])
    if bundle["mode"] != "text" or not trained_on:
        return
    seen = {r.get("extractor") for r in rows if r.get("extractor")}
    if not seen:
        print("  note: input rows carry no extractor field, cannot verify")
        return
    unknown = seen - trained_on
    if not unknown:
        return
    msg = (f"extractor mismatch: input has {sorted(unknown)}, "
           f"model trained on {sorted(trained_on)}")
    if not allow_mismatch:
        raise SystemExit(f"{msg}\n  pass --allow-extractor-mismatch to score anyway")
    print(f"  WARNING: {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a bundle from train_classifier.py")
    ap.add_argument("--input", required=True, help="jsonl to score")
    ap.add_argument("--output", help="jsonl of scores (default: stats only)")
    ap.add_argument("--base-rate", type=float, default=None,
                    help="assumed fraction of the input that is legal, e.g. "
                         "0.001. Prints the precision ceiling the flag rate "
                         "implies at each threshold.")
    ap.add_argument("--allow-extractor-mismatch", action="store_true",
                    help="score even if the input was extracted with a "
                         "different tool than the model was trained on")
    args = ap.parse_args()

    bundle = load_bundle(args.model)
    field = MODES[bundle["mode"]][2]

    rows = list(read_jsonl(args.input))
    check_extractors(bundle, rows, args.allow_extractor_mismatch)

    scored = [r for r in rows if r.get(field)]
    skipped = len(rows) - len(scored)
    if not scored:
        raise SystemExit(f"no rows in {args.input} carry a '{field}' field")
    print(f"\n--- scoring {len(scored)} rows from {args.input}"
          + (f" ({skipped} skipped, no {field})" if skipped else "") + " ---")

    probs = legal_probs(bundle, [r[field] for r in scored])

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r, p in zip(scored, probs):
                out = {"url": r.get("url"), "prob_legal": round(float(p), 6),
                       "flagged": bool(p >= bundle["threshold"])}
                if r.get("label"):
                    out["label"] = r["label"]
                f.write(json.dumps(out) + "\n")
        print(f"wrote {args.output}")

    n = len(probs)
    print(f"\n--- flag rate over {n} rows ---")
    header = f"{'thresh':>7} {'flagged':>9} {'flag rate':>11}"
    if args.base_rate:
        header += f"{'max precision':>15}"
    print(header)
    for t in REPORT_THRESHOLDS:
        k = int((probs >= t).sum())
        rate = k / n
        line = f"{t:>7.2f} {k:>9} {rate:>10.4%}"
        if args.base_rate:
            # every legal page in the input, even if the model caught all of
            # them, is at most base_rate * n of the flagged set
            ceiling = min(1.0, args.base_rate / rate) if rate else float("nan")
            line += f"{ceiling:>14.1%}" if rate else f"{'-':>15}"
        print(line)

    if args.base_rate:
        print(f"\nceiling assumes {args.base_rate:.3%} of the input is legal and "
              f"the model finds all of it.\nReal precision is lower. Hand-label "
              f"a random draw from the flagged set to measure it.")


if __name__ == "__main__":
    main()

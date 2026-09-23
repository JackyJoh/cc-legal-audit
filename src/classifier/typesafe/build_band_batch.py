"""
Builds a labeling batch for the Jev score band 0.75-0.90, the slice just
below the 0.90 cut openweb_precision.py already validated (160/160 legal,
Wilson 95% CI [0.9766, 1.0]).

Why this band. TF-IDF has to drop to ~0.35-0.37 to admit every one of those
160 known-legal pages into a cascade, and at that threshold it lets in far
more candidates than legal pages exist to justify - most of the volume is
TF-IDF false positives, and every one of them is a wasted Jev call. Whether
that's worth paying depends on a question this band answers: does Jev's
precision hold up below 0.90, or does it degrade toward the base rate? If
0.75-0.90 is still near-perfect, a lower Jev accept threshold becomes
possible, which changes what TF-IDF threshold the cascade actually needs -
a looser Jev cut could tolerate a tighter, cheaper TF-IDF cut instead of the
one already measured. If it degrades, 0.90 stays the floor and TF-IDF's low
threshold is the real cost of the cascade, not a solvable inefficiency.

No dedup against any other label file. Per project policy, nothing already
on disk (run_*.jsonl, legal_sample_batch.jsonl, deployment_sample*.jsonl, or
anything derived from them) counts as ground truth for measuring an LLM's
precision - all of it traces through an LLM somewhere. Every URL in this
band goes out with label: "your_label", a placeholder to overwrite with
"legal" or "non_legal", the same convention jev_openweb_kept_full.jsonl uses.

Source: data/labels/jev_openweb_scores.jsonl, already scored, no new Jev
calls and no cost - this only re-slices data already collected.

Outputs:
  data/candidates/jev_openweb_060_090_scores.jsonl   {url, p_legal}
  data/candidates/jev_openweb_060_090_full.jsonl      {url, label: "your_label"}, to hand-label

Usage:
  python src/classifier/typesafe/build_band_batch.py
"""
import json

LO = 0.60
HI = 0.90   # exclusive - jev_openweb_kept_full.jsonl already covers >= 0.90

SCORES_IN   = "data/labels/jev_openweb_scores.jsonl"
SCORES_OUT  = "data/candidates/jev_openweb_060_090_scores.jsonl"
FULL_OUT    = "data/candidates/jev_openweb_060_090_full.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    rows = load_jsonl(SCORES_IN)
    # last line wins per url, matching every other reader of this file
    p_by_url = {}
    for r in rows:
        p_by_url[r["url"]] = r["p_legal"]
    print(f"scored: {len(p_by_url):,}")

    band = sorted(u for u, p in p_by_url.items() if LO <= p < HI)
    print(f"band [{LO}, {HI}): {len(band)} urls")

    with open(SCORES_OUT, "w", encoding="utf-8") as f:
        for u in band:
            f.write(json.dumps({"url": u, "p_legal": p_by_url[u]}) + "\n")
    print(f"  wrote {SCORES_OUT}")

    with open(FULL_OUT, "w", encoding="utf-8") as f:
        for u in band:
            f.write(json.dumps({"url": u, "label": "your_label"}) + "\n")
    print(f"  wrote {FULL_OUT}  (label: \"your_label\" placeholder, hand-label from here)")


if __name__ == "__main__":
    main()

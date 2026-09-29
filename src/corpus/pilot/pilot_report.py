"""
Reports the MinHash pilot from the hand-labeled pair batch.

Per set and Jaccard band: label counts and the share of pairs that are not the
same document, counted two ways (different only; different + version), with
Wilson 95% intervals. Then the legal - control gap, pooled over >= 0.7 (the
main test), with Newcombe 95% intervals. The 0.6-0.7 band is not drawn.

Reads data/pilot/pair_batch.jsonl (labels) and pair_scores.jsonl.
Prints only.
"""
import math
from collections import Counter

from minhash_pairs import BANDS
from sampling import OUT_DIR, load_jsonl

BATCH  = f"{OUT_DIR}/pair_batch.jsonl"
SCORES = f"{OUT_DIR}/pair_scores.jsonl"
LABELS = ("same", "version", "different", "exclude")
READINGS = {"different":           {"different"},
            "different + version": {"different", "version"}}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - half), min(1.0, c + half)


def newcombe(k1, n1, k2, n2):
    """95% interval for p1 - p2 from the two Wilson intervals."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def share(rows, hits):
    """(k, n) over judged rows, excludes dropped."""
    judged = [r for r in rows if r["label"] != "exclude"]
    return sum(1 for r in judged if r["label"] in hits), len(judged)


def main():
    scores = {r["pair"]: r for r in load_jsonl(SCORES)}
    batch = load_jsonl(BATCH)
    bad = [r["pair"] for r in batch if r["label"] and r["label"] not in LABELS]
    if bad:
        print(f"WARNING: unknown labels on pairs {bad}; they are left out")
    rows = [{**scores[r["pair"]], "label": r["label"]} for r in batch if r["label"] in LABELS]
    print(f"{len(rows)} of {len(batch)} pairs labeled")

    print(f"\n  {'set':<9}{'band':>6}{'n':>5}" + "".join(f"{l:>11}" for l in LABELS)
          + f"{'not same':>11}{'95% CI':>17}")
    for name in ("legal", "control"):
        for b in BANDS:
            cell = [r for r in rows if r["set"] == name and r["band"] == b]
            c = Counter(r["label"] for r in cell)
            k, n = share(cell, READINGS["different + version"])
            lo, hi = wilson(k, n)
            ratio = f"{k / n:>11.0%}{f'[{lo:.2f}, {hi:.2f}]':>17}" if n else f"{'-':>11}{'-':>17}"
            print(f"  {name:<9}{b:>6.1f}{len(cell):>5}" + "".join(f"{c[l]:>11}" for l in LABELS) + ratio)

    n_auto = sum(1 for r in batch if r.get("auto"))
    print(f"\n{n_auto} of those are identical-text pairs auto-labeled same")

    for title, bands in (("MAIN TEST, pooled >= 0.7", {0.7, 0.8, 0.9}),):
        print(f"\n{title}: legal - control")
        for reading, hits in READINGS.items():
            k1, n1 = share([r for r in rows if r["set"] == "legal" and r["band"] in bands], hits)
            k2, n2 = share([r for r in rows if r["set"] == "control" and r["band"] in bands], hits)
            if not (n1 and n2):
                print(f"  {reading:<21} not enough labels")
                continue
            d, lo, hi = newcombe(k1, n1, k2, n2)
            print(f"  {reading:<21} legal {k1}/{n1} = {k1 / n1:.0%}   control {k2}/{n2} = {k2 / n2:.0%}"
                  f"   gap {d:+.0%}  [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()

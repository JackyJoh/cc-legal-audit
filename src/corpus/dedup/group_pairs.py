"""
Applies each similarity threshold to find_pairs.py's pairs: groups the pages
connected through pairs at or above it, keeps one page per group, and lists the
rest as removed. The cheap half of deduplication; rerun it freely.

Per threshold (0.6, 0.7, 0.8, 0.9 by default):

  1. pairs      keep the pairs scoring at or above the threshold
  2. groups     pages connected through those pairs form one group, even if
                two of them don't pair directly (A~B and B~C: A, B, C)
  3. survivor   in each group, the page with the lowest survivor_rank
                (minhash.py) is kept; the rest are removed

Each page's rank is fixed, so a page kept at a lower threshold is also kept at
every higher one: only the threshold changes what is removed.

Usage:
  python src/corpus/dedup/group_pairs.py --out data/dedup

Reads (--out): pairs.npy and pages.npy, from find_pairs.py

Outputs (--out):
  removed_0.8.npy   one file per threshold: each removed page's id, and the
                    id of the page kept in its group (kept_id)
Prints per threshold: pages removed per bucket, the number of groups and the
biggest, groups mixing legal and general pages, and legal pages removed while a
general page was kept.
"""
import argparse

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from find_pairs import BUCKETS, save
from minhash import survivor_rank

THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
LEGAL = BUCKETS.index("legal")
REMOVED = np.dtype([("id", "<i8"), ("kept_id", "<i8")])


# one threshold: each page's group number, and the positions (in pages) of the
# removed pages and of the page kept in each one's group
def group(pages, pairs, t):
    p = pairs[pairs["score"] >= t]
    n = len(pages)
    a = np.searchsorted(pages["id"], p["id_a"])
    b = np.searchsorted(pages["id"], p["id_b"])
    links = coo_matrix((np.ones(len(p), np.int8), (a, b)), shape=(n, n))
    _, label = connected_components(links, directed=False)   # pages with no pair: a group of one

    # sort by group, then rank: each group's first page is its survivor
    order = np.lexsort((survivor_rank(pages["id"]), label))
    first = np.r_[True, label[order][1:] != label[order][:-1]]
    survivor = order[np.maximum.accumulate(np.where(first, np.arange(n), 0))]
    return label, order[~first], survivor[~first]


# per-threshold summary, per bucket and for groups mixing buckets
def report(t, pages, label, removed, kept):
    legal = pages["bucket"] == LEGAL
    sizes = np.bincount(label)
    legal_in = np.bincount(label, weights=legal)
    mixed = int(((legal_in > 0) & (legal_in < sizes)).sum())
    for code, name in enumerate(BUCKETS):
        total = int((pages["bucket"] == code).sum())
        gone = int((pages["bucket"][removed] == code).sum())
        share = f"{gone / total:.1%}" if total else "-"
        print(f"  {t:<11}{name:<9}{total:>12,}{gone:>12,}{share:>9}")
    lost_to_general = int((legal[removed] & ~legal[kept]).sum())
    print(f"  {'':<11}{int((sizes > 1).sum()):,} groups (biggest {int(sizes.max(initial=0)):,} pages), "
          f"{mixed:,} mixing legal and general; {lost_to_general:,} legal pages removed "
          f"with a general page kept")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", default="data/dedup", help="find_pairs.py's --out folder")
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    args = ap.parse_args()

    pages = np.load(f"{args.out}/pages.npy")
    pages = pages[np.argsort(pages["id"])]          # sorted, so ids can be looked up
    pairs = np.load(f"{args.out}/pairs.npy")
    print(f"{len(pages):,} pages, {len(pairs):,} pairs\n")
    print(f"  {'threshold':<11}{'bucket':<9}{'pages':>12}{'removed':>12}{'share':>9}")

    for t in sorted(args.thresholds):
        label, removed, kept = group(pages, pairs, t)
        rows = np.empty(len(removed), REMOVED)
        rows["id"], rows["kept_id"] = pages["id"][removed], pages["id"][kept]
        save(f"{args.out}/removed_{t}.npy", rows)
        report(t, pages, label, removed, kept)


if __name__ == "__main__":
    main()
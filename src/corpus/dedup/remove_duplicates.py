"""
Applies each similarity threshold to score_pairs.py's pairs: groups the pages
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
  python src/corpus/dedup/remove_duplicates.py --out data/source

Reads (--out/dedup/): pairs.npy and pages.npy, from score_pairs.py

Outputs (--out/dedup/):
  removed_0.8.npy   one file per threshold: each removed page's id, and the
                    id of the page kept in its group (kept_id)
Prints one table, a row per threshold: pages removed per bucket (count and
share), the number of groups and the biggest, groups mixing legal and general
pages, and legal pages removed while a general page was kept.
"""
import argparse
import time
from datetime import timedelta

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from score_pairs import BUCKETS, save
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


# one threshold's row of the summary table: removed per bucket (count and
# share), groups, biggest group, groups mixing buckets, legal lost to general
def report_row(t, pages, label, removed, kept):
    legal = pages["bucket"] == LEGAL
    sizes = np.bincount(label)
    legal_in = np.bincount(label, weights=legal)
    mixed = int(((legal_in > 0) & (legal_in < sizes)).sum())
    row = [f"{t}"]
    for code in range(len(BUCKETS)):
        total = int((pages["bucket"] == code).sum())
        gone = int((pages["bucket"][removed] == code).sum())
        row.append(f"{gone:,}  ({gone / total:.1%})" if total else "-")
    lost_to_general = int((legal[removed] & ~legal[kept]).sum())
    return row + [f"{int((sizes > 1).sum()):,}", f"{int(sizes.max(initial=0)):,}",
                  f"{mixed:,}", f"{lost_to_general:,}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", default="data/source", help="source_pages.py's --out folder")
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    args = ap.parse_args()
    start = time.monotonic()                     # wall clock for the run, printed at the end
    dedup = f"{args.out}/dedup"

    pages = np.load(f"{dedup}/pages.npy")
    pages = pages[np.argsort(pages["id"])]          # sorted, so ids can be looked up
    pairs = np.load(f"{dedup}/pairs.npy")
    console = Console()
    counts = ", ".join(f"{int((pages['bucket'] == c).sum()):,} {b}" for c, b in enumerate(BUCKETS))
    console.print(f"{len(pages):,} pages ({counts}), {len(pairs):,} pairs\n")

    table = Table("threshold", *(f"{b} removed" for b in BUCKETS), "groups", "biggest",
                  "mixed groups", "legal lost to general")
    for col in table.columns:
        col.justify, col.no_wrap = "right", True
    for t in sorted(args.thresholds):
        label, removed, kept = group(pages, pairs, t)
        rows = np.empty(len(removed), REMOVED)
        rows["id"], rows["kept_id"] = pages["id"][removed], pages["id"][kept]
        save(f"{dedup}/removed_{t}.npy", rows)
        table.add_row(*report_row(t, pages, label, removed, kept))
    console.print(table)
    console.print(f"took {timedelta(seconds=round(time.monotonic() - start))}")


if __name__ == "__main__":
    main()
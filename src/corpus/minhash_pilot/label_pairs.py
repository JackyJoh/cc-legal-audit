"""
Terminal viewer for labeling data/minhash_pilot/pair_batch.jsonl.

Shows one unlabeled pair at a time (URLs, word counts, opening text, word diff)
and writes the label back to the file after every answer, so you can quit and
resume. Only pair_batch.jsonl is read; scores are never shown.

  s  same       copy, or only formatting / site chrome differs
  v  version    same document, different version (amended, introduced vs. enrolled)
  d  different  different documents that look alike from shared structure
  x  exclude    can't tell (e.g. extraction broke)
  b  back       relabel the previous pair
  h  help       print the labeling rules
  q  quit

Labeling rules (fixed 2026-09-28, early in labeling; the pairs already
labeled then fit them) are in RULES below and print at startup and on h.

Usage:
  python src/corpus/minhash_pilot/label_pairs.py [--relabel]

Outputs (data/minhash_pilot/):
  pair_batch.jsonl   the same file, with each pair's label filled in

Next:
  python src/corpus/minhash_pilot/pilot_report.py
"""
import argparse
import json
import os
import sys
import textwrap

from sampling import OUT_DIR, load_jsonl

BATCH  = f"{OUT_DIR}/pair_batch.jsonl"
LABELS = {"s": "same", "v": "version", "d": "different", "x": "exclude"}
WIDTH  = 100

# the study measures lost content (topic entropy), not legal identity, so a
# label asks: if dedup dropped one page, what content would be lost?
RULES = """
LABELING RULES: label by what dedup would lose if it dropped one page
  same       operative text identical; only identifiers / metadata differ:
             section, bill or case numbers, agency or court name, authority
             citations, register / history notes, dates, URLs, punctuation,
             capitalization, site chrome
  version    SAME instrument (same identifier) at different times; operative
             text changed by amendment, or introduced vs. enrolled
             (dated URLs, amendment citations)
  different  operative text differs beyond identifiers: a page has provisions,
             holdings, definitions or other substance the other lacks, even if
             most of the page is shared template or chrome
  exclude    can't judge: extraction broke, page empty or chrome-only, diff
             cut off where it matters
tie-breakers
  1. ignore volume: one unique operative sentence is enough for different;
     shared chrome alone never is
  2. ignore identifiers: a different name or number on the same text is not
     new content
  3. same vs different unsure: same if the unique text is only metadata;
     different only if you can point to a unique operative sentence
  4. containment: one page is a part of the other (page-1 vs FullText of
     the same instrument, same date) -> same; it's one document split up,
     not two documents sharing boilerplate
examples
  Canada consolidations dated 20041215 vs 20070420       version
  VA budget amendment, HB1600 vs SB800, text identical   same
  VA APA definitions, Dept of Law vs Auctioneers Board   same
  Canada Act 2022_14, FullText vs page-1                 same
  WI DWD 81.09 vs 81.10 (different provisions)           different
  Boost / Tenor pages sharing only the site template     different
"""


# write to a temp file then swap it in, so a crash mid-save can't corrupt the batch
def save(rows):
    tmp = BATCH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, BATCH)


def wrap(text, indent=""):
    return textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=" " * len(indent) + "  ")


# print one pair: where it is, what the page is, and only what differs between the two
def show(r, done, total):
    print("\n" + "=" * WIDTH)
    print(f"pair {r['pair']}   ({done}/{total} labeled)"
          + (f"   current label: {r['label']}" if r["label"] else ""))
    print("=" * WIDTH)
    print(f"A  {r['url_a']}")
    print(f"B  {r['url_b']}")
    print(f"words: A {r['words_a']:,}   B {r['words_b']:,}")
    print("\nstarts:")
    print(wrap(r["start"], "  "))
    print(f"\ndiff ({len(r['diff'])} shown):" if r["diff"] else "\ndiff: none (identical text)")
    for i, h in enumerate(r["diff"], 1):
        print(wrap(h, f"  {i:>2}. "))


def prompt():
    while True:
        a = input("\n[s]ame  [v]ersion  [d]ifferent  e[x]clude  [b]ack  [h]elp  [q]uit > ").strip().lower()
        if a in LABELS or a in ("b", "q"):
            return a
        if a == "h":
            print(RULES)
            continue
        print("  s, v, d, x, b, h or q")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--relabel", action="store_true", help="go through all pairs, not just unlabeled ones")
    args = ap.parse_args()

    # diffs contain non-cp1252 characters (e.g. the ∅ marker); Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = load_jsonl(BATCH)
    order = [i for i, r in enumerate(rows) if args.relabel or not r["label"]]
    if not order:
        print("all pairs are labeled (use --relabel to revisit)")
        return

    print(RULES)
    pos = 0
    while pos < len(order):
        r = rows[order[pos]]
        show(r, sum(1 for x in rows if x["label"]), len(rows))
        a = prompt()
        if a == "q":
            break
        if a == "b":
            pos = max(0, pos - 1)
            continue
        r["label"] = LABELS[a]
        save(rows)
        pos += 1

    done = sum(1 for x in rows if x["label"])
    print(f"\n{done}/{len(rows)} labeled -> {BATCH}")
    if done == len(rows):
        print("Next: python src/corpus/minhash_pilot/pilot_report.py")


if __name__ == "__main__":
    main()

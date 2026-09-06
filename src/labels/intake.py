"""
Merges all labeling-agent output into one training file, tagging every row
with which sourcing pass produced it:

  original  the original 2990-URL candidate batch (rule-based prefilter
            hits + raw random URLs + synthetic homepage negatives), scoped
            to data/candidates/candidates.jsonl. Carries all three
            corrections (see below).
  target    the error-driven +500 batch that targeted the false-negative
            domains and URL-shape gaps found after training on the
            original batch, scoped to data/candidates/targeted_batch.jsonl.
            Carries the definition-popup correction (see below).
  cl        the CourtListener-derived host sample (fetch_cl_hostnames.py's
            court + legislature domains, sampled via fetch_cl_urls.py),
            scoped to data/candidates/host_sample_batch.jsonl.

Each batch is merged independently with the same logic: a labeled row whose
URL isn't in that batch's own candidate file is dropped as contamination, a
duplicate label for the same URL is deduped, a conflicting duplicate label
is logged instead of silently picked, and a label beats a skip for the same
URL. The three results are then concatenated into one file.

Corrections. The labeling prompt originally allowed a page that "directly
links to" a filing to count as legal, but Common Crawl only captures a
page's own HTML, not what it links to, so a landing page with no document
text on the page itself is a false positive. LINK_ONLY_CORRECTIONS
(18 URLs, original batch) is that failure, hand-confirmed by rationale
text. A second worker applied one templated rationale across a whole
Justice Laws Canada URL family without checking each page;
INDEX_PAGE_CORRECTIONS (16 URLs, original batch) is that failure, confirmed
by fetching 5 of them directly. is_definition_popup (311 rows, original and
targeted batches) is the third, described at its definition below. All
three are corrected to non_legal here rather than discarded, since they're
useful hard negatives: right domain, right vocabulary, wrong page type.

Output: data/processed/labeled_urls.jsonl, one clean {"url", "label",
"confidence", "rationale", "run_id", "labeled_at", "source"} object per
line.
"""
import json
import glob
import os

LABELS_DIR = "data/labels"
OUTPUT_FILE = "data/processed/labeled_urls.jsonl"

LINK_ONLY_CORRECTIONS = {
    "https://laws-lois.justice.gc.ca/eng/acts/O-2.7/index.html",
    "https://laws-lois.justice.gc.ca/eng/acts/C-10.15/",
    "https://www.laws-lois.justice.gc.ca/eng/acts/A-0.6/",
    "https://lawfilesext.leg.wa.gov/Law/WACArchive/2021/pdf/WAC%20%2036%20%20TITLE/WAC%20%2036%20%20%20TITLE/",
    "https://lois-laws.justice.gc.ca/eng/acts/M-7.01/PITIndex.html",
    "https://www.law.cornell.edu/cfr/text/49/part-272",
    "https://laws-lois.justice.gc.ca/eng/acts/F-4/",
    "https://law.lis.virginia.gov/admincode/title12/agency30/chapter306/",
    "https://laws.justice.gc.ca/eng/regulations/SOR-2019-251/?wbdisable=true",
    "https://www.law.cornell.edu/uscode/text/16/chapter-58/subchapter-VII",
    "https://www.wvlegislature.gov/bill_status/bills_history.cfm?input=4210&year=2006&sessiontype=RS&btype=bill",
    "https://www.law.cornell.edu/cfr/text/5/part-3101",
    "https://www.judiciary.uk/prevention-of-future-death-reports/christopher-hart/",
    "https://www.judiciary.uk/prevention-of-future-death-reports/michael-worrall/",
    "https://www.judiciary.uk/judgments/settle-group-v-hodgson/",
    "https://www.judiciary.uk/judgments/mermaids-v-the-charity-commission-for-england-wales/",
    "https://www.judiciary.uk/judgments/barclays-bank-v-dylan-and-others/",
    "https://www.judiciary.uk/judgments/sana-musharraf-v-r/",
}

INDEX_PAGE_CORRECTIONS = {
    "https://laws-lois.justice.gc.ca/eng/regulations/SOR-2007-123/PITIndex.html",
    "https://laws-lois.justice.gc.ca/eng/regulations/SOR-92-446/PITIndex.html",
    "https://www.laws.justice.gc.ca/eng/regulations/SOR-97-332/PITIndex.html",
    "https://lois-laws.justice.gc.ca/eng/regulations/SOR-86-1121/index.html",
    "https://laws-lois.justice.gc.ca/eng/acts/G-11.55/PITIndex.html",
    "https://www.laws-lois.justice.gc.ca/eng/regulations/SOR-2012-209/index.html",
    "https://www.laws.justice.gc.ca/eng/regulations/SOR-88-529/",
    "https://www.laws-lois.justice.gc.ca/eng/regulations/SI-97-48/index.html",
    "https://www.laws.justice.gc.ca/eng/regulations/SOR-2002-336/20060626/P1TT3xt3.html",
    "https://laws.justice.gc.ca/eng/regulations/SOR-2006-355/",
    "https://laws-lois.justice.gc.ca/eng/regulations/SOR-94-120/",
    "https://www.laws.justice.gc.ca/eng/regulations/SOR-2005-35/FullText.html",
    "https://www.laws-lois.justice.gc.ca/eng/regulations/SOR-2011-78/PITIndex.html",
    "https://laws-lois.justice.gc.ca/eng/regulations/SOR-90-247/index.html",
    "https://lois-laws.justice.gc.ca/eng/regulations/SOR-61-507/index.html",
    "https://www.laws.justice.gc.ca/eng/acts/G-11.8/?wbdisable=true",
}

# Cornell LII's inline definition popup: an iframe modal (note the fixed
# width/height/iframe query params on every one) showing a single defined
# term in ~45-65 words, above a "Source" line that links out to the actual
# CFR section on a different page. Non_legal on three separate counts under
# the definition in prompts/legal_url_labeling_task.md - it is a fragment
# rather than a document's text, its own HTML links to the real text
# instead of containing it (the same failure LINK_ONLY_CORRECTIONS covers),
# and it is UI chrome rather than a publication. Confirmed by fetching two
# of them directly. At that length they also fall below the Gopher quality
# filter the pipeline applies before classification, so the classifier
# would be learning to chase pages that never reach it.
#
# A predicate rather than a URL set: 311 rows across two batches match, and
# any future batch sampling law.cornell.edu will pull more.
#
# This family was labeled both ways, and the split fell along batch lines
# rather than page content: the original batch called it non_legal 87 to
# 44, then every one of the targeted batch's 149 cornell.edu URLs was this
# same popup and every one was labeled legal. The false-negative analysis
# that sourced that batch had read the original batch's non_legal labels as
# a cornell.edu "URL-shape gap" and gone looking for more of the shape, so
# the targeted pass was amplifying a labeling disagreement rather than
# fixing a model error. Re-run that analysis now this is settled.
#
# Covers both variants of the widget, which are the only two URL shapes
# under /definitions/ in any batch: index.php (the CFR popup, 280 rows) and
# uscode.php (the US Code popup, 31 rows). uscode.php was checked
# separately by fetching one row from each side of its label split - 29 USC
# 1002(34) "individual account plan" (labeled legal) and 49 USC 1136(h)(3)
# "passenger list" (labeled non_legal) - and they are the same page down to
# the layout: one defined term, 50-65 words, a "Source" line hyperlinking
# to the full section elsewhere. Its split is worse than index.php's, too,
# falling inside individual workers rather than between batches (w2 3-3,
# w5 2-2, w6 2-2, w4 1-6), so one agent in one session called the same
# widget both ways.
def is_definition_popup(url):
    return "law.cornell.edu/definitions/" in url


DEFINITION_POPUP_NOTE = (
    "[corrected: inline definition-popup widget - one defined term, links "
    "out to the actual section text rather than containing it]"
)

# name, candidate file, run-file globs, skipped-file globs, corrections
# (each correction is a (url_set_or_predicate, note) pair applied only
# within this batch; a predicate is tested against every labeled URL)
BATCHES = [
    {
        "source": "original",
        "candidates_file": "data/candidates/candidates.jsonl",
        "run_globs": ["run_2026-08-27-w*.jsonl", "run_2026-08-27-manual.jsonl"],
        "skipped_globs": ["skipped_2026-08-27-w*.jsonl"],
        "corrections": [
            (LINK_ONLY_CORRECTIONS,
             "[corrected: page itself has no legal text, only links to one]"),
            (INDEX_PAGE_CORRECTIONS,
             "[corrected: landing/index page, no regulatory text present on page itself]"),
            (is_definition_popup, DEFINITION_POPUP_NOTE),
        ],
    },
    {
        "source": "target",
        "candidates_file": "data/candidates/targeted_batch.jsonl",
        "run_globs": ["run_2026-08-27-targeted-w*.jsonl"],
        "skipped_globs": ["skipped_2026-08-27-targeted-w*.jsonl"],
        "corrections": [
            (is_definition_popup, DEFINITION_POPUP_NOTE),
        ],
    },
    {
        "source": "cl",
        "candidates_file": "data/candidates/host_sample_batch.jsonl",
        "run_globs": ["run_2026-08-31-hostsample-w*.jsonl"],
        "skipped_globs": ["skipped_2026-08-31-hostsample-w*.jsonl"],
        "corrections": [],
    },
]


def load_candidates(path):
    urls = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                urls.add(json.loads(line)["url"])
    return urls


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def glob_all(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(LABELS_DIR, pattern)))
    return sorted(paths)


def merge_batch(batch):
    candidates = load_candidates(batch["candidates_file"])

    labeled_by_url = {}
    contamination = []
    same_url_conflicts = []
    for path in glob_all(batch["run_globs"]):
        for row in load_jsonl(path):
            url = row["url"]
            if url not in candidates:
                contamination.append(row)
                continue
            if url in labeled_by_url:
                prior = labeled_by_url[url]
                if prior["label"] != row["label"]:
                    same_url_conflicts.append((prior, row))
                continue
            labeled_by_url[url] = row

    skipped_urls = set()
    for path in glob_all(batch["skipped_globs"]):
        for row in load_jsonl(path):
            if row["url"] in candidates:
                skipped_urls.add(row["url"])
    skipped_only = skipped_urls - set(labeled_by_url)

    correction_counts = []
    for match, note in batch["corrections"]:
        correction_urls = ([u for u in labeled_by_url if match(u)]
                           if callable(match) else match)
        n = 0
        for url in correction_urls:
            row = labeled_by_url.get(url)
            if row and row["label"] == "legal":
                row["label"] = "non_legal"
                row["rationale"] = (row.get("rationale") or "") + " " + note
                n += 1
        correction_counts.append(n)

    return {
        "n_candidates": len(candidates),
        "labeled_by_url": labeled_by_url,
        "skipped_only": skipped_only,
        "contamination": contamination,
        "same_url_conflicts": same_url_conflicts,
        "correction_counts": correction_counts,
    }


def report(batch, result):
    labeled = result["labeled_by_url"]
    legal = sum(1 for r in labeled.values() if r["label"] == "legal")
    non_legal = sum(1 for r in labeled.values() if r["label"] == "non_legal")
    other = len(labeled) - legal - non_legal
    covered = len(labeled) + len(result["skipped_only"])

    print(f"\n--- {batch['source']}  ({batch['candidates_file']}) ---")
    print(f"Candidates            : {result['n_candidates']}")
    print(f"Clean labeled         : {len(labeled)}  ({legal} legal, {non_legal} non_legal, {other} other)")
    for (_, note), n in zip(batch["corrections"], result["correction_counts"]):
        print(f"Corrections           : {n} (legal -> non_legal, {note})")
    print(f"Skipped (unlabeled)   : {len(result['skipped_only'])}")
    print(f"Dropped, not in batch : {len(result['contamination'])}")
    for row in result["contamination"]:
        print(f"  - {row['url']}  (from run_id={row.get('run_id')})")
    print(f"Same-URL label conflicts: {len(result['same_url_conflicts'])}")
    for prior, row in result["same_url_conflicts"]:
        print(f"  - {prior['url']}: {prior['run_id']}={prior['label']} vs {row['run_id']}={row['label']}")
    print(f"Coverage              : {covered} / {result['n_candidates']} candidates accounted for")


def main():
    combined = []
    seen_elsewhere = {}
    for batch in BATCHES:
        result = merge_batch(batch)
        report(batch, result)
        for url, row in result["labeled_by_url"].items():
            if url in seen_elsewhere:
                print(f"\nWARNING: {url} labeled in both "
                      f"'{seen_elsewhere[url]}' and '{batch['source']}' batches")
            seen_elsewhere[url] = batch["source"]
            combined.append({
                "url": row["url"],
                "label": row["label"],
                "confidence": row.get("confidence"),
                "rationale": row.get("rationale"),
                "run_id": row.get("run_id"),
                "labeled_at": row.get("labeled_at"),
                "source": batch["source"],
            })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")

    print(f"\n=== combined ===")
    for batch in BATCHES:
        n = sum(1 for r in combined if r["source"] == batch["source"])
        print(f"  {batch['source']:<8} {n}")
    print(f"Total labeled URLs    : {len(combined)}")
    print(f"Written               : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

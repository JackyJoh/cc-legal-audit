"""
One-time merge of the raw labeling agent output (data/labels/run_*.jsonl,
data/labels/skipped_*.jsonl) into a single clean training file.

Handles the mess that six parallel workers actually produced:
  - a handful of URLs one worker fetched that were never in the candidate
    batch (it strayed off its assigned slice), dropped and logged
  - a few URLs labeled twice by the same worker (identical or near-identical
    calls both times), deduped to one
  - any URL that ended up both labeled and skipped, resolved in favor of
    the label since a label means the content was actually read

Output: data/processed/labeled_urls.jsonl, one clean {"url", "label",
"confidence", "rationale", "run_id", "labeled_at"} object per line.
"""
import json
import glob
import os

CANDIDATES_FILE = "data/candidates/candidates.jsonl"
LABELS_DIR = "data/labels"
OUTPUT_FILE = "data/processed/labeled_urls.jsonl"

# Manual correction pass: the labeling prompt's definition allowed a page
# that "directly links to" a filing/opinion to count as legal, on the
# assumption the linked document would still be reachable. But Common
# Crawl only captures the fetched page's own HTML, not whatever a link on
# that page points to, so a landing/index page that merely links out to a
# PDF has no legal text on the page itself and is useless (or actively
# misleading) for the downstream BERTopic/entropy analysis. These 18 were
# manually confirmed by hand-scanning the legal-labeled set and flagged
# by rationale text ("links to", "linked pdf", etc.) for containing no
# actual document text on the page. Corrected to non_legal here, since
# they're real, useful hard negatives (right domain, wrong page type),
# not noise to discard.
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

# Second correction pass: Justice Laws Canada (laws-lois.justice.gc.ca and
# its mirrors) uses index.html / PITIndex.html / FullText.html / a bare
# act-root URL for landing/metadata pages that link out to the actual text
# in separate HTML/XML/PDF files, and numbered pages (page-N.html) for the
# actual text. One worker applied an identical templated rationale
# ("Full-text landing page of a specific Canadian federal regulation or
# act...") across this whole URL family without distinguishing the two,
# an instance of the exact per-URL-verification failure the labeling
# prompt was written to prevent. Confirmed by directly fetching 5 of these
# and finding metadata/links only, no regulatory text, in all 5.
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


def main():
    candidates = load_candidates(CANDIDATES_FILE)

    labeled_by_url = {}
    contamination = []
    same_url_conflicts = []
    for path in sorted(glob.glob(os.path.join(LABELS_DIR, "run_*.jsonl"))):
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
    for path in sorted(glob.glob(os.path.join(LABELS_DIR, "skipped_*.jsonl"))):
        for row in load_jsonl(path):
            if row["url"] in candidates:
                skipped_urls.add(row["url"])

    skipped_only = skipped_urls - set(labeled_by_url)

    corrected = 0
    for url in LINK_ONLY_CORRECTIONS:
        row = labeled_by_url.get(url)
        if row and row["label"] == "legal":
            row["label"] = "non_legal"
            row["rationale"] = (row.get("rationale") or "") + \
                " [corrected: page itself has no legal text, only links to one]"
            corrected += 1

    index_corrected = 0
    for url in INDEX_PAGE_CORRECTIONS:
        row = labeled_by_url.get(url)
        if row and row["label"] == "legal":
            row["label"] = "non_legal"
            row["rationale"] = (row.get("rationale") or "") + \
                " [corrected: landing/index page, no regulatory text present on page itself]"
            index_corrected += 1

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in labeled_by_url.values():
            f.write(json.dumps({
                "url": row["url"],
                "label": row["label"],
                "confidence": row.get("confidence"),
                "rationale": row.get("rationale"),
                "run_id": row.get("run_id"),
                "labeled_at": row.get("labeled_at"),
            }) + "\n")

    legal = sum(1 for r in labeled_by_url.values() if r["label"] == "legal")
    non_legal = sum(1 for r in labeled_by_url.values() if r["label"] == "non_legal")
    other = len(labeled_by_url) - legal - non_legal

    print(f"Candidates            : {len(candidates)}")
    print(f"Clean labeled         : {len(labeled_by_url)}  ({legal} legal, {non_legal} non_legal, {other} other)")
    print(f"Link-only corrections : {corrected} (legal -> non_legal)")
    print(f"Index-page corrections: {index_corrected} (legal -> non_legal)")
    print(f"Skipped (unlabeled)   : {len(skipped_only)}")
    print(f"Dropped, not in batch : {len(contamination)}")
    for row in contamination:
        print(f"  - {row['url']}  (from run_id={row.get('run_id')})")
    print(f"Same-URL label conflicts: {len(same_url_conflicts)}")
    for prior, row in same_url_conflicts:
        print(f"  - {prior['url']}: {prior['run_id']}={prior['label']} vs {row['run_id']}={row['label']}")
    covered = len(labeled_by_url) + len(skipped_only)
    print(f"Coverage              : {covered} / {len(candidates)} candidates accounted for")
    print(f"Written               : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

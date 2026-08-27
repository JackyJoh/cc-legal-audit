"""
Merge of the targeted-batch labeling agent output (data/labels/run_2026-08-27-
targeted-w*.jsonl, data/labels/skipped_2026-08-27-targeted-w*.jsonl) into a
single clean file, same pattern as intake.py but scoped to the error-driven
+500 batch (data/candidates/targeted_batch.jsonl) instead of the original
candidate batch.

No manual link-only/index-page corrections here: this batch was labeled
under the already-tightened definition in prompts/legal_url_labeling_task.md
(page must contain the text itself, not just link to it), so the systematic
drift that required correction in intake.py shouldn't be present. Run the
same grep-by-rationale-keyword audit before trusting this blindly.

Output: data/processed/targeted_labeled_urls.jsonl, one clean {"url",
"label", "confidence", "rationale", "run_id", "labeled_at"} object per line.
"""
import json
import glob
import os

CANDIDATES_FILE = "data/candidates/targeted_batch.jsonl"
LABELS_DIR = "data/labels"
RUN_GLOB = "run_2026-08-27-targeted-w*.jsonl"
SKIPPED_GLOB = "skipped_2026-08-27-targeted-w*.jsonl"
OUTPUT_FILE = "data/processed/targeted_labeled_urls.jsonl"


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
    for path in sorted(glob.glob(os.path.join(LABELS_DIR, RUN_GLOB))):
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
    for path in sorted(glob.glob(os.path.join(LABELS_DIR, SKIPPED_GLOB))):
        for row in load_jsonl(path):
            if row["url"] in candidates:
                skipped_urls.add(row["url"])

    skipped_only = skipped_urls - set(labeled_by_url)

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

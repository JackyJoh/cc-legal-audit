"""
Error-driven follow-up to fetch_candidate_urls.py. Not a random sample -
targets the specific failure modes found in scratch_false_negatives.py's
held-out false-negative analysis:

  - Tier A (thin domains): vermont.gov, judiciary.uk, wa.gov, utcourts.gov,
    wvlegislature.gov missed 100% of their (3-5 example) held-out legal
    URLs. Broad domain pull, no path filter - they just need volume.
  - Tier B (diversity-gap domains): cornell.edu, justice.gc.ca,
    virginia.gov are well-represented but still miss ~25-50% of held-out
    legal URLs, because training skews toward whichever URL shape the
    original prefilter happened to grab most. Path-pattern-targeted pulls
    aimed at the specific underrepresented shapes identified in the FN
    list (cornell's /definitions/ and /regulations/ paths, justice.gc.ca's
    page-N.html/P1TT3xt3.html/section-sched paths, virginia's
    admincodefull/ paths), explicitly skipping the already-well-covered
    shapes (cornell /uscode/text/, /cfr/text/; virginia vacode/).

No separate hard-negative bucket - these pulls aren't filtered by the old
legal-likely classifier, so they naturally include non-legal pages on the
same domains/paths (index pages, menus, etc.), same as the original raw
pool did.

Every URL already present in data/candidates/candidates.jsonl (already
labeled) is dropped before writing, so nothing gets relabeled.

Output: data/candidates/targeted_batch.jsonl, one {"url": ..., "hint": ...}
per line. hint is "thin_domain:<domain>" or "diversity_gap:<domain>",
triage-only, not ground truth - same convention as build_label_batch.py.
"""
import json
import os
import time
import random
import boto3
from dotenv import load_dotenv

load_dotenv()

# config
SNAPSHOT        = "CC-MAIN-2026-12"
SEED            = 42
ATHENA_DB       = "ccindex"
CANDIDATES_FILE = "data/candidates/candidates.jsonl"
OUTPUT_FILE     = "data/candidates/targeted_batch.jsonl"

OUTPUT_LOCATION = os.environ['ATHENA_OUTPUT_LOCATION']

BASE_WHERE = f"crawl = '{SNAPSHOT}' AND subset = 'warc' AND content_languages = 'eng'"

# Tier A: thin domains, 100% held-out miss rate, only 3-5 test examples each.
THIN_DOMAINS = [
    "vermont.gov", "judiciary.uk", "wa.gov", "utcourts.gov", "wvlegislature.gov",
]
TIER_A_LIMIT_PER_DOMAIN = 50

# Tier B: well-represented domains, still missing due to URL-shape skew.
# limit weighted roughly by held-out FN count (cornell 27, gc.ca 12, virginia 8).
TIER_B_QUERIES = [
    {
        "name": "cornell.edu",
        "hint": "diversity_gap:cornell.edu",
        "limit": 150,
        "where": (
            "url_host_registered_domain = 'cornell.edu' "
            "AND (url_path LIKE '%/definitions/%' OR url_path LIKE '%/regulations/%')"
        ),
    },
    {
        "name": "justice.gc.ca",
        "hint": "diversity_gap:justice.gc.ca",
        "limit": 75,
        "where": (
            "url_host_registered_domain = 'justice.gc.ca' "
            "AND (url_path LIKE '%page-%.html' OR url_path LIKE '%P1TT3xt3.html' "
            "OR url_path LIKE '%section-sched%')"
        ),
    },
    {
        "name": "virginia.gov",
        "hint": "diversity_gap:virginia.gov",
        "limit": 75,
        "where": (
            "url_host_registered_domain = 'virginia.gov' "
            "AND url_path LIKE '%admincodefull%'"
        ),
    },
]


def run_query(client, sql):
    resp = client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={'Database': ATHENA_DB},
        ResultConfiguration={'OutputLocation': OUTPUT_LOCATION}
    )
    qid = resp['QueryExecutionId']

    while True:
        state = (client.get_query_execution(QueryExecutionId=qid)
                 ['QueryExecution']['Status']['State'])
        if state in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
            break
        time.sleep(2)

    if state != 'SUCCEEDED':
        raise RuntimeError(f"Athena query {state}: {qid}")

    rows, first = [], True
    for page in client.get_paginator('get_query_results').paginate(QueryExecutionId=qid):
        for row in page['ResultSet']['Rows']:
            if first:
                first = False
                continue  # skip header row
            rows.append(row['Data'][0]['VarCharValue'])
    return rows


def load_candidate_urls(path):
    urls = set()
    if not os.path.exists(path):
        return urls
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                urls.add(json.loads(line)['url'])
    return urls


def main():
    client = boto3.client('athena', region_name='us-east-1')
    random.seed(SEED)

    already_labeled = load_candidate_urls(CANDIDATES_FILE)
    print(f"Already-labeled candidates (excluded): {len(already_labeled)}")

    results = []  # list of (url, hint)
    seen = set()

    print("\n--- Tier A: thin domains ---")
    for domain in THIN_DOMAINS:
        sql = f"""
            SELECT url FROM ccindex
            WHERE {BASE_WHERE}
              AND url_host_registered_domain = '{domain}'
            LIMIT {TIER_A_LIMIT_PER_DOMAIN}
        """
        urls = run_query(client, sql)
        hint = f"thin_domain:{domain}"
        n_new = 0
        for u in urls:
            if u in already_labeled or u in seen:
                continue
            seen.add(u)
            results.append((u, hint))
            n_new += 1
        print(f"  {domain:<20} queried={len(urls):>4}  new={n_new:>4}")

    print("\n--- Tier B: diversity-gap path patterns ---")
    for q in TIER_B_QUERIES:
        sql = f"""
            SELECT url FROM ccindex
            WHERE {BASE_WHERE}
              AND {q['where']}
            LIMIT {q['limit']}
        """
        urls = run_query(client, sql)
        n_new = 0
        for u in urls:
            if u in already_labeled or u in seen:
                continue
            seen.add(u)
            results.append((u, q["hint"]))
            n_new += 1
        print(f"  {q['name']:<20} queried={len(urls):>4}  new={n_new:>4}")

    random.shuffle(results)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for url, hint in results:
            f.write(json.dumps({"url": url, "hint": hint}) + "\n")

    print(f"\nTotal new URLs: {len(results)}")
    print(f"Written       : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

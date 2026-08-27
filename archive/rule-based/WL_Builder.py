"""
Discover candidate legal domains in a CC snapshot via Athena.

Queries the columnar index for every registered domain whose host name contains
a legal keyword, grouped by registered domain and ordered by page count.
Manually triage the output into a whitelist of primary legal sources.

Prerequisites:
  1. ccindex table registered in Athena (see CC docs)
  2. ATHENA_OUTPUT_LOCATION set in .env
  3. MSCK REPAIR TABLE ccindex run recently enough to include the snapshot

"""
import os
import time
import boto3
from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
SNAPSHOT     = "CC-MAIN-2026-12"   # research snapshot
ATHENA_DB    = "ccindex"
MIN_PAGES    = 1000               # drop one-off obscure domains
OUTPUT_FILE  = "wl_candidates.txt"
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_LOCATION = os.environ['ATHENA_OUTPUT_LOCATION']

# Broad discovery keywords — wider than runtime classifier on purpose.
DISCOVERY_KEYWORDS = [
    'court', 'legal', 'law', 'judicial', 'judiciary', 'statute',
    'attorney', 'lawyer', 'litigation', 'jurisprudence',
    'tribunal', 'legislat', 'appellate',
]


def build_query():
    like_clauses = "\n     OR ".join(
        f"url_host_name LIKE '%{kw}%'" for kw in DISCOVERY_KEYWORDS
    )
    return f"""
        SELECT
            url_host_name AS domain,
            COUNT(*) AS pages
        FROM ccindex
        WHERE crawl = '{SNAPSHOT}'
          AND subset = 'warc'
          AND content_languages LIKE '%eng%'
          AND ({like_clauses})
        GROUP BY url_host_name
        HAVING COUNT(*) >= {MIN_PAGES}
        ORDER BY pages DESC
    """


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
                continue  # skip header
            domain = row['Data'][0]['VarCharValue']
            pages = int(row['Data'][1]['VarCharValue'])
            rows.append((domain, pages))
    return rows


def main():
    client = boto3.client('athena', region_name='us-east-1')
    sql = build_query()

    print(f"Snapshot : {SNAPSHOT}")
    print(f"Keywords : {DISCOVERY_KEYWORDS}")
    print(f"Running discovery query (30–90s)...")

    results = run_query(client, sql)

    print(f"\nFound {len(results)} candidate domains.")
    print(f"Writing to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# Candidate legal domains from {SNAPSHOT}\n")
        f.write(f"# Triage each: keep in whitelist if primary legal source, else delete the line.\n")
        f.write(f"# Format: pages\tdomain\n\n")
        for domain, pages in results:
            f.write(f"{pages}\t{domain}\n")

    print("Done. Review the file and remove non-legal domains to build whitelist.")


if __name__ == "__main__":
    main()

"""
Sample URLs from CC columnar index via Athena, classify them, and print positives
+ a negative sample for manual precision/recall review.

Prerequisites:
  1. Register ccindex table in Athena (follow CC docs — CREATE DATABASE + CREATE EXTERNAL TABLE)
  2. Set ATHENA_OUTPUT_LOCATION in .env to an S3 bucket you own (Athena writes results there)

Cost: ~$0.50 per run (scans one crawl partition ~100GB at $5/TB).
"""
import os
import time
import random
import boto3
from dotenv import load_dotenv
from URL_Classifier import classify, InWhitelist

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────
SNAPSHOT     = "CC-MAIN-2026-12"   # swap to "CC-MAIN-2026-17" for the real run
SEED         = 42
N_SAMPLE     = 5000
N_NEG_REVIEW = 75
ATHENA_DB    = "ccindex"
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_LOCATION = os.environ['ATHENA_OUTPUT_LOCATION']


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


def main():
    client = boto3.client('athena', region_name='us-east-1')

    sql = f"""
        SELECT url
        FROM ccindex TABLESAMPLE BERNOULLI(1)
        WHERE crawl = '{SNAPSHOT}'
          AND subset = 'warc'
          AND content_languages = 'eng'
        LIMIT {N_SAMPLE}
    """

    print(f"Snapshot : {SNAPSHOT}")
    print(f"Running Athena query (30–60s)...")

    urls = run_query(client, sql)
    random.seed(SEED)
    random.shuffle(urls)

    positives = [u for u in urls if classify(u)]
    negatives = [u for u in urls if not classify(u)]

    from urllib.parse import urlparse
    wl_hits = [u for u in positives if InWhitelist(urlparse(u).netloc)]
    kw_hits = [u for u in positives if not InWhitelist(urlparse(u).netloc)]

    random.seed(SEED)
    neg_sample = random.sample(negatives, min(N_NEG_REVIEW, len(negatives)))

    # ── precision review ──────────────────────────────────────────────────────
    print(f"\n=== POSITIVES ({len(positives)}) — mark each TP or FP ===")
    for i, u in enumerate(positives, 1):
        tag = "[WL]" if InWhitelist(urlparse(u).netloc) else "[KW]"
        print(f"  {i:3}. {tag} {u}")

    # ── KW hits only (for FP review) ─────────────────────────────────────────
    print(f"\n=== KW HITS ONLY ({len(kw_hits)}) ===")
    for i, u in enumerate(kw_hits, 1):
        print(f"  {i:3}. {u}")

    # ── recall review ─────────────────────────────────────────────────────────
    print(f"\n=== NEGATIVE SAMPLE ({len(neg_sample)} of {len(negatives)}) — mark any that ARE legal (FN) ===")
    for i, u in enumerate(neg_sample, 1):
        print(f"  {i:3}. {u}")

    # ── raw counts ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Total sampled  : {len(urls)}")
    print(f"Positives      : {len(positives)}  ({len(wl_hits)} WL, {len(kw_hits)} KW)  [{100*len(positives)/len(urls):.1f}%]")
    print(f"Negatives      : {len(negatives)}  ({len(neg_sample)} reviewed, {len(negatives)-len(neg_sample)} unreviewed)")
    print()
    print("Precision = TP / (TP + FP)   [count from positives list]")
    print("Recall    = TP_est / (TP_est + FN_est)")
    print(f"  FN in reviewed sample → scale: FN_est = FN_found × ({len(negatives)} / {len(neg_sample)})")


if __name__ == "__main__":
    main()

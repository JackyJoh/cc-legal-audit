"""
Samples raw URLs from the CC columnar index via Athena and writes them to a
raw pool file. No classification happens here, this just sources candidate
URLs. Run build_label_batch.py to turn this raw pool into the actual
batch handed to the labeling agent (prefilter hits + raw random + homepage
negatives).

Prerequisites:
  1. Register ccindex table in Athena (follow CC docs - CREATE DATABASE + CREATE EXTERNAL TABLE)
  2. Set ATHENA_OUTPUT_LOCATION in .env to an S3 bucket you own (Athena writes results there)

Cost: about $0.50 per run (scans one crawl partition, ~100GB at $5/TB), and
that's independent of how many rows you request, so pulling a bigger pool
here is effectively free.
"""
import json
import os
import time
import random
import boto3
from dotenv import load_dotenv

load_dotenv()

# config
SNAPSHOT     = "CC-MAIN-2026-12"
SEED         = 42
N_SAMPLE     = 600000
ATHENA_DB    = "ccindex"
OUTPUT_FILE  = "data/candidates/raw_pool.jsonl"

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
    print(f"Running Athena query (30-60s)...")

    urls = run_query(client, sql)
    random.seed(SEED)
    random.shuffle(urls)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for url in urls:
            f.write(json.dumps({"url": url}) + "\n")

    print(f"Sampled  : {len(urls)} URLs")
    print(f"Written  : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

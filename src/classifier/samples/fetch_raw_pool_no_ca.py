"""
Fresh Athena pull for the deployment-validation follow-up: the first
21-URL sample (sample_deployment_validation.py, threshold 0.85) turned out
to be 100% lois.justice.gc.ca - nothing else in that 600k pool cleared the
threshold. This pulls a different, larger portion of the crawl with .ca
domains excluded at the SQL level, so the next validation sample can't
overlap with the first one and can't be Canadian by construction.

Cost is the same ~$0.50/run as fetch_candidate_urls.py (scans one crawl
partition regardless of row count), so pulling more rows here is free -
sized up to 1.5M (vs. the original 600k) since confident non-.ca hits
appear to be rare.
"""
import json
import os
import time
import random
import boto3
from dotenv import load_dotenv

load_dotenv()

SNAPSHOT     = "CC-MAIN-2026-12"
SEED         = 43  # different from fetch_candidate_urls.py's 42, on purpose
N_SAMPLE     = 1500000
ATHENA_DB    = "ccindex"
OUTPUT_FILE  = "data/candidates/raw_pool_no_ca.jsonl"

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
                continue
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
          AND url_host_tld != 'ca'
        LIMIT {N_SAMPLE}
    """

    print(f"Snapshot : {SNAPSHOT}  (excluding .ca)")
    print(f"Running Athena query...")

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

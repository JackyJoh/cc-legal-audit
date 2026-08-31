"""
Shared Athena helper.

fetch_candidate_urls.py and fetch_targeted_urls.py each carry their own
single-column run_query. The host-count and per-host-sample stages need
multiple columns back, plus the bytes-scanned figure so a run's cost is
visible instead of guessed at, so that logic lives here once.
"""
import os
import time

import boto3

REGION      = "us-east-1"
ATHENA_DB   = "ccindex"
# Athena on-demand pricing, $5 per TB scanned. Only used to print an estimate.
USD_PER_TB  = 5.0
_TB         = 1024 ** 4


def client():
    return boto3.client("athena", region_name=REGION)


def run_query(athena, sql, database=ATHENA_DB, poll=2.0, quiet=False):
    """Run a query to completion, return (rows, stats).

    rows is a list of dicts keyed by the result column names. Null cells come
    back as None rather than being silently dropped, which matters because
    content_languages and content_mime_detected are both nullable.
    """
    output_location = os.environ["ATHENA_OUTPUT_LOCATION"]
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )
    qid = resp["QueryExecutionId"]

    while True:
        execution = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = execution["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(poll)

    if state != "SUCCEEDED":
        reason = execution["Status"].get("StateChangeReason", "no reason given")
        raise RuntimeError(f"Athena query {state} ({qid}): {reason}")

    stats = execution.get("Statistics", {})
    scanned = stats.get("DataScannedInBytes", 0)
    millis = stats.get("EngineExecutionTimeInMillis", 0)

    rows = []
    columns = None
    for page in athena.get_paginator("get_query_results").paginate(QueryExecutionId=qid):
        for row in page["ResultSet"]["Rows"]:
            values = [cell.get("VarCharValue") for cell in row["Data"]]
            if columns is None:
                columns = values  # first row of the first page is the header
                continue
            rows.append(dict(zip(columns, values)))

    stats = {
        "query_execution_id": qid,
        "scanned_bytes": scanned,
        "scanned_gb": scanned / (1024 ** 3),
        "est_cost_usd": scanned / _TB * USD_PER_TB,
        "seconds": millis / 1000.0,
        "n_rows": len(rows),
    }
    if not quiet:
        print(f"  [{qid[:8]}] {stats['seconds']:.1f}s  "
              f"{stats['scanned_gb']:.2f} GB scanned  "
              f"~${stats['est_cost_usd']:.2f}  {len(rows)} rows")
    return rows, stats


def sql_in_list(values):
    """Render a Python iterable as a SQL IN-list literal.

    Single quotes are doubled rather than escaped with a backslash, which is
    the SQL standard and what Trino expects. Hostnames should never contain
    one, but building the literal correctly beats trusting that.
    """
    escaped = sorted({str(v).replace("'", "''") for v in values})
    return ", ".join(f"'{v}'" for v in escaped)

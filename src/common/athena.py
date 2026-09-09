"""
Shared Athena helper: get a client, run a query to completion, get rows back.

client()       boto3 Athena client for the fixed region/database.
run_query()    Runs SQL against the `ccindex` table, retrying transient
               AWS-side faults (S3 throttling, split errors) with backoff.
               Returns (rows, stats): rows is a list of {column: value}
               dicts (nulls preserved as None, not dropped), stats carries
               bytes/GB scanned, an estimated dollar cost, elapsed time and
               row count.
sql_in_list()  Renders a Python iterable as a quoted SQL IN-list literal.

fetch_candidate_urls.py and fetch_targeted_urls.py each carried their own
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


# S3 rate limiting and transient engine faults come back as a FAILED query
# carrying a reason string, not as a boto3 error, so boto3's own retry layer
# never sees them and they can only be recognized by matching the reason. A
# retry re-scans the data and so is charged again, which is why the list stays
# narrow: only faults that clear on their own. Sweeping several crawls back to
# back is the usual way to trip the throttling one.
TRANSIENT_REASONS = (
    "HIVE_S3_THROTTLING",
    "SlowDown",
    "HIVE_CANNOT_OPEN_SPLIT",
    "INTERNAL_ERROR",
)
MAX_ATTEMPTS = 4


def client():
    return boto3.client("athena", region_name=REGION)


def _execute(athena, sql, database, output_location, poll):
    """Start one query and poll it to a terminal state."""
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
            return qid, state, execution
        time.sleep(poll)


def run_query(athena, sql, database=ATHENA_DB, poll=2.0, quiet=False):
    """Run a query to completion, return (rows, stats).

    rows is a list of dicts keyed by the result column names. Null cells come
    back as None rather than being silently dropped, which matters because
    content_languages and content_mime_detected are both nullable.

    Transient AWS-side failures are retried with backoff. A cancelled query and
    a broken query both fail immediately, since neither improves on a retry.
    """
    output_location = os.environ["ATHENA_OUTPUT_LOCATION"]

    for attempt in range(MAX_ATTEMPTS):
        qid, state, execution = _execute(athena, sql, database, output_location, poll)
        if state == "SUCCEEDED":
            break
        reason = execution["Status"].get("StateChangeReason", "no reason given")
        if attempt == MAX_ATTEMPTS - 1 or not any(r in reason for r in TRANSIENT_REASONS):
            raise RuntimeError(f"Athena query {state} ({qid}): {reason}")
        wait = 10 * 2 ** attempt
        print(f"  [{qid[:8]}] {state}, retry in {wait}s: {reason}")
        time.sleep(wait)

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

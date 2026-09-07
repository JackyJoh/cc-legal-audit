"""
Fetches the Common Crawl page text for a list of URLs.

Usage:
  python src/corpus/fetch_warc_text.py                     # the labeled set
  python src/corpus/fetch_warc_text.py --input X --output Y

Written for the labeled URLs, but any jsonl with a "url" field works, which
is what lets the deployment sample and the label set go through identical
extraction. That matters more than it sounds: the model learns from whatever
the extractor produces, so text prepared a different way is a different
input, and comparisons across it mean nothing.

How it works, in two stages so the Athena spend happens once:

  1. Locate. Ask the Common Crawl index where each URL's capture lives:
     which archive file, which byte offset, how many bytes. The URL list is
     too long for a single query (Athena caps a query string at 262KB), so it
     goes out in chunks, each one a separate partition scan costing about
     $0.14. The answers are cached to a pointers file, so this stage runs
     once per URL list and never again. An input that already carries pointer
     columns skips the stage entirely.
  2. Fetch and extract. Range-request exactly those bytes from
     data.commoncrawl.org over plain HTTP, so no AWS credentials and no S3
     egress charge, then run trafilatura over the HTML.

Extraction settings are load-bearing and pinned. deduplicate is set off
explicitly: the option drops repeated segments, and an extractor that quietly
dedupes its own input would confound the fuzzy-dedup study this corpus feeds.
include_tables and favor_recall are on because statutes live in tables and
nested lists, where losing subsection (b)(2) is a worse error than keeping an
extra paragraph.

Every input URL gets an output row. Failures keep the row and carry a
skip_reason with a null text instead of disappearing, so counts always
reconcile against the input, and html_bytes and text_chars are both recorded
so pages where extraction ate the content stay visible. Re-running resumes:
rows that failed for a transient reason (throttling, timeouts) are retried,
rows that are settled (not in the crawl, extracted to nothing) are not.

The text output is large and gitignored. The pointers file is small, is meant
to be committed, and regenerates the text with no further Athena spend.
"""
import argparse
import io
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import trafilatura
from dotenv import load_dotenv
from warcio.archiveiterator import ArchiveIterator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "samples"))
from athena import client, run_query, sql_in_list  # noqa: E402

load_dotenv()

SNAPSHOT = "CC-MAIN-2026-12"
DEFAULT_INPUT = "data/processed/labeled_urls.jsonl"
DEFAULT_POINTERS = "data/processed/warc_pointers.jsonl"
DEFAULT_OUTPUT = "data/processed/labeled_text.jsonl"

# columns that make an input row self-locating: if all three are present the
# index lookup is already done and stage 1 is skipped
POINTER_COLS = ("warc_filename", "warc_record_offset", "warc_record_length")

# Athena caps a query string at 262,144 chars; stay well under it so the
# chunker never has to reason about the exact overhead of the SQL around
# the IN-list.
MAX_QUERY_CHARS = 200_000
# data.commoncrawl.org throttles with 403/503 under load. 16 workers with
# immediate retries lost 31% of a full run to throttling; backing off and
# halving the concurrency trades a few minutes for the whole corpus.
N_WORKERS = 6
FETCH_RETRIES = 4
BACKOFF_BASE = 1.5
TIMEOUT = 60
# reasons worth another attempt on a later run; anything else is settled
TRANSIENT = ("http 403", "http 503", "http 500", "http 502", "http 504",
             "Timeout", "Connection", "Chunked", "fetch failed")
# a handful of captures are enormous; truncation keeps one page from
# dominating the vectorizer's vocabulary
MAX_TEXT_CHARS = 500_000
CC_BASE = "https://data.commoncrawl.org/"

# Pinned deliberately - trafilatura 1.x and 2.x return different text for the
# same HTML, so the version is recorded on every row rather than assumed.
EXTRACTOR = f"trafilatura-{trafilatura.__version__}"
EXTRACT_OPTS = dict(
    output_format="txt",
    include_tables=True,     # statutes are frequently laid out in tables
    include_comments=False,
    favor_recall=True,       # prefer keeping a subsection over dropping it
    deduplicate=False,       # never let the extractor dedupe the dedup study
)


def needs_retry(row):
    """True for a row that failed for a reason another run might survive."""
    if row.get("text"):
        return False
    reason = row.get("skip_reason") or ""
    return any(t in reason for t in TRANSIENT)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def chunk_urls(urls):
    """Split the URL list so each rendered IN-list stays under the cap."""
    chunks, current, size = [], [], 0
    for u in urls:
        cost = len(u) + 4  # quotes, comma, space
        if current and size + cost > MAX_QUERY_CHARS:
            chunks.append(current)
            current, size = [], 0
        current.append(u)
        size += cost
    if current:
        chunks.append(current)
    return chunks


def fetch_pointers(urls):
    """Ask Athena where each URL's capture lives in the crawl archives."""
    athena = client()
    chunks = chunk_urls(urls)
    print(f"  {len(urls)} URLs -> {len(chunks)} quer{'y' if len(chunks) == 1 else 'ies'} "
          f"(~$0.50 each, one partition scan apiece)")

    by_url, total_cost = {}, 0.0
    for i, chunk in enumerate(chunks, 1):
        sql = f"""
            SELECT url,
                   warc_filename,
                   warc_record_offset,
                   warc_record_length,
                   fetch_status,
                   content_mime_detected,
                   content_languages
            FROM ccindex
            WHERE crawl = '{SNAPSHOT}'
              AND subset = 'warc'
              AND url IN ({sql_in_list(chunk)})
        """
        print(f"  chunk {i}/{len(chunks)} ({len(chunk)} URLs)")
        rows, stats = run_query(athena, sql)
        total_cost += stats["est_cost_usd"]
        for r in rows:
            # a URL can be captured more than once in a crawl; keep the
            # lowest (filename, offset) so re-runs pick the same record
            key = (r["warc_filename"], int(r["warc_record_offset"]))
            prev = by_url.get(r["url"])
            if prev is None or key < (prev["warc_filename"], int(prev["warc_record_offset"])):
                by_url[r["url"]] = r

    print(f"  matched {len(by_url)}/{len(urls)} URLs, ~${total_cost:.2f} total")
    return by_url


def fetch_record(pointer):
    """Range-GET one WARC record and return its raw HTTP payload bytes."""
    offset = int(pointer["warc_record_offset"])
    length = int(pointer["warc_record_length"])
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    url = CC_BASE + pointer["warc_filename"]

    last = None
    for attempt in range(FETCH_RETRIES):
        if attempt:
            # jittered exponential backoff: retrying a throttle immediately
            # is what turned rate limiting into a third of the corpus
            time.sleep(BACKOFF_BASE ** attempt + random.random())
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
            if resp.status_code not in (200, 206):
                last = f"http {resp.status_code}"
                continue
            for record in ArchiveIterator(io.BytesIO(resp.content)):
                if record.rec_type == "response":
                    return record.content_stream().read(), None
            return None, "no response record"
        except Exception as exc:  # network, gzip, or WARC parse failure
            last = f"{type(exc).__name__}: {exc}"
    return None, last or "fetch failed"


def build_row(item):
    """Turn one input URL into an output row, success or not.

    label and source pass through when the input has them and are null when
    it does not, so an unlabeled crawl sample produces the same row shape as
    the label set and can be scored by the same code.
    """
    label_row, pointer = item
    base = {
        "url": label_row["url"],
        "label": label_row.get("label"),
        "source": label_row.get("source"),
        "extractor": EXTRACTOR,
        "text": None,
        "html_bytes": None,
        "text_chars": None,
        "skip_reason": None,
    }

    if pointer is None:
        base["skip_reason"] = "not_in_crawl"
        return base

    base["fetch_status"] = pointer.get("fetch_status")
    base["mime"] = pointer.get("content_mime_detected")
    base["languages"] = pointer.get("content_languages")

    html, err = fetch_record(pointer)
    if html is None:
        base["skip_reason"] = err
        return base
    base["html_bytes"] = len(html)

    try:
        text = trafilatura.extract(html, **EXTRACT_OPTS)
    except Exception as exc:
        base["skip_reason"] = f"extract error: {type(exc).__name__}"
        return base

    if not text or not text.strip():
        base["skip_reason"] = "empty extraction"
        return base

    base["text"] = text[:MAX_TEXT_CHARS]
    base["text_chars"] = len(base["text"])
    return base


def resolve_pointers(rows, pointers_file):
    """Stage 1: get each URL's location in the crawl archives.

    Three ways to get there, cheapest first: the input already carries the
    columns, the cached pointers file already covers the URL, or Athena has to
    be asked. The cache is consulted per URL rather than all-or-nothing, so
    adding a batch of labels queries only the new URLs instead of paying to
    re-locate every URL the file already holds.

    A URL that Athena has already been asked about and did not match stays
    missing from the cache, so it is asked about again on the next run. That
    is the cost of keeping the file to matches only; it is small because a URL
    that is not in the crawl is dropped from the label set anyway.
    """
    if rows and all(all(r.get(c) for c in POINTER_COLS) for r in rows):
        print("  input already carries pointers, skipping Athena")
        return {r["url"]: r for r in rows}

    pointers = {}
    if pointers_file and os.path.exists(pointers_file):
        pointers = {r["url"]: r for r in load_jsonl(pointers_file)}
        print(f"  cached: {pointers_file} ({len(pointers)} rows)")

    missing = [r["url"] for r in rows if r["url"] not in pointers]
    if not missing:
        print("  every URL already located, skipping Athena")
        return pointers

    print(f"  {len(missing)} URL(s) not in the cache, querying the index")
    pointers.update(fetch_pointers(missing))
    if pointers_file:
        os.makedirs(os.path.dirname(pointers_file) or ".", exist_ok=True)
        with open(pointers_file, "w", encoding="utf-8") as f:
            for url in sorted(pointers):
                f.write(json.dumps(pointers[url]) + "\n")
        print(f"  wrote {pointers_file} ({len(pointers)} rows)")
    return pointers


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help="jsonl with a 'url' field per row")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="jsonl of extracted text, one row per input URL")
    ap.add_argument("--pointers", default=DEFAULT_POINTERS,
                    help="where the index lookup is cached, so it runs once "
                         "per URL list")
    args = ap.parse_args()

    INPUT_FILE, OUTPUT_FILE, POINTERS_FILE = args.input, args.output, args.pointers

    labeled = load_jsonl(INPUT_FILE)
    print(f"--- {len(labeled)} URLs from {INPUT_FILE} ---")

    print("\n1. WARC pointers")
    pointers = resolve_pointers(labeled, POINTERS_FILE)

    print("\n2. Page text")
    # A previous run's throttled rows are worth another attempt; a row that
    # simply isn't in the crawl, or that extracted to nothing, is settled.
    have = {}
    if os.path.exists(OUTPUT_FILE):
        have = {r["url"]: r for r in load_jsonl(OUTPUT_FILE)}
        retryable = sum(1 for r in have.values() if needs_retry(r))
        print(f"  resuming: {len(have)} rows on disk, {retryable} worth retrying")

    todo = [(r, pointers.get(r["url"])) for r in labeled
            if r["url"] not in have or needs_retry(have[r["url"]])]
    if not todo:
        print("  nothing to do")
    else:
        lock = threading.Lock()
        done_n = 0
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            for row in pool.map(build_row, todo):
                with lock:
                    have[row["url"]] = row
                    done_n += 1
                    if done_n % 250 == 0:
                        print(f"  {done_n}/{len(todo)}")

        # rewritten whole rather than appended, so a retried URL replaces its
        # earlier failure instead of appearing twice
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
            for url in sorted(have):
                out.write(json.dumps(have[url], ensure_ascii=False) + "\n")

    rows = load_jsonl(OUTPUT_FILE)
    ok = [r for r in rows if r["text"]]
    print(f"\n  {len(rows)} rows, {len(ok)} with text, "
          f"{len(rows) - len(ok)} skipped")
    if any(r.get("label") for r in ok):
        print(f"  legal with text   : {sum(1 for r in ok if r['label'] == 'legal')}")
        print(f"  non_legal with text: {sum(1 for r in ok if r['label'] == 'non_legal')}")
    if len(rows) != len(labeled):
        print(f"  WARNING: {len(rows)} rows != {len(labeled)} input URLs")


if __name__ == "__main__":
    main()

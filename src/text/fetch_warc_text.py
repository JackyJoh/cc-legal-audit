"""
Pulls the Common Crawl page text for every URL in labeled_urls.jsonl, so the
same labels can train a content classifier instead of a URL-string one.

Two stages, so the Athena spend happens once:

  1. warc_pointers.jsonl  where each labeled URL sits in the crawl archives
     (filename, byte offset, length, plus fetch_status and mime). The URL
     list renders to ~349KB of SQL literal against Athena's 262KB query cap,
     so it goes out in chunks - each chunk is its own partition scan at
     roughly $0.50, which is why the result is cached to disk.
  2. labeled_text.jsonl   the extracted text, fetched by HTTP range request
     against data.commoncrawl.org: no AWS credentials, no S3 egress charge.

Extraction is trafilatura with deduplicate switched off explicitly. It
already defaults off in 2.2.0, but the option strips repeated segments, and
an extractor that quietly dedupes its own input would confound the fuzzy-dedup
study this corpus exists to support. include_tables and favor_recall are on
for a related reason: statutes and regulations live in tables and nested
lists, and dropping subsection (b)(2) is not the same class of error as
dropping a paragraph from a blog post.

Every input URL gets an output row. Failures carry a skip_reason and a null
text rather than vanishing, so the row count always reconciles against
labeled_urls.jsonl, and both html_bytes and text_chars are recorded so pages
where extraction ate 99% of the content are visible afterwards. Both stages
resume by skipping URLs already written.

Output: data/processed/labeled_text.jsonl. That file runs to ~100MB and is
gitignored; warc_pointers.jsonl is the small reproducibility record and is
meant to be committed, since it regenerates the text without touching Athena
again.
"""
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sourcing"))
from athena import client, run_query, sql_in_list  # noqa: E402

load_dotenv()

SNAPSHOT = "CC-MAIN-2026-12"
LABELED_FILE = "data/processed/labeled_urls.jsonl"
POINTERS_FILE = "data/processed/warc_pointers.jsonl"
OUTPUT_FILE = "data/processed/labeled_text.jsonl"

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
    """Turn one labeled URL into an output row, success or not."""
    label_row, pointer = item
    base = {
        "url": label_row["url"],
        "label": label_row["label"],
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


def main():
    labeled = load_jsonl(LABELED_FILE)
    print(f"--- {len(labeled)} labeled URLs from {LABELED_FILE} ---")

    print("\n1. WARC pointers")
    if os.path.exists(POINTERS_FILE):
        pointers = {r["url"]: r for r in load_jsonl(POINTERS_FILE)}
        print(f"  reusing {POINTERS_FILE} ({len(pointers)} rows) - delete it to re-query")
    else:
        pointers = fetch_pointers([r["url"] for r in labeled])
        os.makedirs(os.path.dirname(POINTERS_FILE), exist_ok=True)
        with open(POINTERS_FILE, "w", encoding="utf-8") as f:
            for url in sorted(pointers):
                f.write(json.dumps(pointers[url]) + "\n")
        print(f"  wrote {POINTERS_FILE}")

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
    print(f"  legal with text   : {sum(1 for r in ok if r['label'] == 'legal')}")
    print(f"  non_legal with text: {sum(1 for r in ok if r['label'] == 'non_legal')}")
    if len(rows) != len(labeled):
        print(f"  WARNING: {len(rows)} rows != {len(labeled)} labeled URLs")


if __name__ == "__main__":
    main()

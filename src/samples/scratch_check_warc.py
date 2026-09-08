"""
Scratch check: pull the raw WARC record for one URL directly, to see exactly
what Common Crawl captured. Not part of the pipeline - prints raw bytes, does
not run extraction.

Inlines the pointer-lookup/fetch logic from fetch_warc_text.py rather than
importing it, since that module pulls in trafilatura, which isn't needed here.
"""
import io
import random
import sys
import time

import requests
from dotenv import load_dotenv
from warcio.archiveiterator import ArchiveIterator

from athena import client, run_query, sql_in_list

load_dotenv()

SNAPSHOT = "CC-MAIN-2026-12"
CC_BASE = "https://data.commoncrawl.org/"
FETCH_RETRIES = 4
BACKOFF_BASE = 1.5
TIMEOUT = 60


def fetch_pointer(url):
    sql = f"""
        SELECT url, warc_filename, warc_record_offset, warc_record_length,
               fetch_status, content_mime_detected, content_languages
        FROM ccindex
        WHERE crawl = '{SNAPSHOT}' AND subset = 'warc'
          AND url IN ({sql_in_list([url])})
    """
    rows, _ = run_query(client(), sql)
    return rows[0] if rows else None


def fetch_record(pointer):
    offset = int(pointer["warc_record_offset"])
    length = int(pointer["warc_record_length"])
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    url = CC_BASE + pointer["warc_filename"]

    last = None
    for attempt in range(FETCH_RETRIES):
        if attempt:
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
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
    return None, last or "fetch failed"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else \
        "https://www.mass.gov/info-details/commonwealth-critical-incident-stress-management-cism-program"

    pointer = fetch_pointer(url)
    if pointer is None:
        print("Not found in the crawl index (CC-MAIN-2026-12).")
        return

    print(f"fetch_status: {pointer.get('fetch_status')}")
    print(f"mime: {pointer.get('content_mime_detected')}")
    print(f"languages: {pointer.get('content_languages')}")
    print(f"warc_filename: {pointer.get('warc_filename')}")
    print(f"offset/length: {pointer.get('warc_record_offset')}/{pointer.get('warc_record_length')}")

    payload, err = fetch_record(pointer)
    if payload is None:
        print(f"fetch failed: {err}")
        return

    print(f"\npayload bytes: {len(payload)}")
    head = payload[:2000]
    print("\n--- first 2000 bytes (raw) ---")
    print(head.decode("utf-8", "replace"))


if __name__ == "__main__":
    main()

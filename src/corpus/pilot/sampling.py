"""
Shared pieces for the pilot samplers: site naming, eligibility, and the
per-site random page draw.
"""
import json
import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from athena import sql_in_list  # noqa: E402

SNAPSHOT = "CC-MAIN-2026-12"
SEED     = "42"
OUT_DIR  = "data/pilot"

HTML_MIMES = ("'text/html'", "'application/xhtml+xml'")
ELIGIBLE = ("fetch_status = 200 "
            f"AND content_mime_detected IN ({', '.join(HTML_MIMES)}) "
            "AND content_languages = 'eng'")
SITE_SQL = r"regexp_replace(url_host_name, '^www\.', '')"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def site_of(url):
    """Host minus a leading www."""
    host = urlparse(url).hostname or ""
    return host[4:] if host.startswith("www.") else host


def spellings(site):
    return {site, "www." + site}


def pages_sql(quota):
    """Random eligible pages from each site, up to its quota, with WARC pointers."""
    values = ", ".join(f"('{s.replace(chr(39), chr(39) * 2)}', {q})"
                       for s, q in sorted(quota.items()))
    hosts = sorted({h for s in quota for h in spellings(s)})
    return f"""
        WITH quota(site, q) AS (VALUES {values}),
        pages AS (
            SELECT url, {SITE_SQL} AS site,
                   warc_filename, warc_record_offset, warc_record_length,
                   fetch_status, content_mime_detected, content_languages,
                   row_number() OVER (
                       PARTITION BY {SITE_SQL}
                       ORDER BY xxhash64(to_utf8(concat(url, '{SEED}')))) AS rn
            FROM ccindex
            WHERE crawl = '{SNAPSHOT}'
              AND subset = 'warc'
              AND {ELIGIBLE}
              AND url_host_name IN ({sql_in_list(hosts)})
        )
        SELECT p.url, p.site, p.warc_filename, p.warc_record_offset,
               p.warc_record_length, p.fetch_status, p.content_mime_detected,
               p.content_languages
        FROM pages p JOIN quota q ON p.site = q.site
        WHERE p.rn <= q.q
    """


def one_capture_per_url(rows):
    """Lowest (filename, offset) per URL, as in fetch_warc_text.py."""
    by_url = {}
    for r in rows:
        key = (r["warc_filename"], int(r["warc_record_offset"]))
        prev = by_url.get(r["url"])
        if prev is None or key < (prev["warc_filename"], int(prev["warc_record_offset"])):
            by_url[r["url"]] = r
    return [by_url[u] for u in sorted(by_url)]

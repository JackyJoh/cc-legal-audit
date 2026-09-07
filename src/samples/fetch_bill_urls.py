"""
Turns the bill sections from fetch_register_sources.py into a batch of real
URLs for the labeling agent.

Open States said which part of each legislature's site holds its bills. This
asks Common Crawl what is actually in those sections and samples from it, so
every URL handed to the agent is a page the crawl really captured. Nothing
is taken from Open States itself, which keeps the labeled set on the same
distribution the classifier will meet at inference.

How it works:

  1. Read the sections file. Each row is a site plus a path pattern:

       www.kslegislature.gov    /li/%/measures/%

     Patterns backed by fewer than MIN_RECORDS bills are dropped, since one
     Open States record pointing somewhere is usually a stray link.

  2. Run one Athena query, covering all the sites at once. It walks every
     page the crawl captured on those sites and sorts each into one of two
     piles, by whether its path matches that site's pattern:

       in    /li/b2025_26/measures/hb2001/     matches
       out   /li/committees/schedule/2026/     does not

     The query returns how many pages are in each pile, and returns the
     piles shuffled. The shuffle comes from sorting on a hash of the URL
     plus a fixed seed, which looks random but lands the same way every
     run, so re-running this script draws the same sample.

  3. Drop any site with fewer than MIN_ELIGIBLE pages in its "in" pile. A
     site the crawler barely reached cannot supply enough rows to be worth
     labeling, and it could never reach the 5 legal rows the
     leave-one-domain-out evaluation needs before it will score a
     publisher at all.

  4. From each surviving site take a fixed quota off the front of each pile
     (--in-per-site and --out-per-site). The piles are already shuffled, so
     taking the first N is a random draw. Every site gets the same quota
     rather than a share of its size, so a deeply-crawled state cannot
     dominate the batch the way justice.gc.ca dominates the existing label
     set.

Why both piles get sampled, not just "in": every legislature has to appear
in training under both labels. If every page from kslegislature.gov were
legal, a model could score well by memorising the domain instead of reading
the text, which is exactly how the URL classifier failed.

Neither pile is a label. The agent judges every page and both piles come
back mixed: "in" holds bill text but also the index pages that list it under
the same path, and "out" is mostly committee business and news but sometimes
statutes.

Output: data/candidates/bill_sample_batch.jsonl, in the {"url", "hint"} shape
intake.py and the labeling prompt expect. The hint names the stream, never
the section, so the agent cannot infer an expected answer from it.
"""
import argparse
import collections
import hashlib
import json
import os

import tldextract
from dotenv import load_dotenv

from athena import client, run_query, sql_in_list

load_dotenv()

SNAPSHOT = "CC-MAIN-2026-12"
SEED = "42"
SOURCES_FILE = "data/candidates/register_sources_bills.jsonl"
OUTPUT_FILE = "data/candidates/bill_sample_batch.jsonl"
BATCH_ID = "bill-sample-v1"

# Below this a legislature cannot support a clean sample, matching the
# threshold fetch_cl_urls.py settled on for the same reason.
MIN_ELIGIBLE = 200
# A section attested by a single Open States record is usually a one-off
# (a stray link, a redirect), not where that state files its bills.
MIN_RECORDS = 3
# Rows pulled per stream before deduping against earlier batches, so a
# legislature overlapping those batches still reaches its quota.
OVERFETCH = 4

# Must match the eligibility used everywhere else in the project. xhtml is
# HTML: matching text/html alone drops whole statute publishers.
HTML_MIMES = ("'text/html'", "'application/xhtml+xml'")

# URLs already handed to the labeling agent, excluded so nothing is relabeled.
EXISTING_BATCHES = [
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
    "data/candidates/host_sample_batch.jsonl",
]

_extract = tldextract.TLDExtract(suffix_list_urls=())


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_urls(paths):
    urls = set()
    for path in paths:
        if os.path.exists(path):
            urls.update(r["url"] for r in load_jsonl(path))
    return urls


def rank(*parts):
    """Deterministic order. blake2b rather than the builtin hash(), which is
    salted per process and would reshuffle the sample on every run."""
    return hashlib.blake2b((SEED + "|".join(parts)).encode("utf-8"),
                           digest_size=8).digest()


def sql_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def build_sql(sections, k_in, k_out):
    """One scan, both streams.

    A page counts as inside when its host matches and its path matches one of
    that host's section patterns. Everything else on those hosts is outside.
    row_number over a seeded hash pre-ranks within each host and stream, so
    the sample is drawn in the database rather than by pulling every row back.
    """
    by_host = collections.defaultdict(list)
    for s in sections:
        by_host[s["host"]].append(s["pattern"])

    clauses = []
    for host, patterns in sorted(by_host.items()):
        likes = " OR ".join(f"url_path LIKE {sql_quote(p)}" for p in sorted(set(patterns)))
        clauses.append(f"(url_host_name = {sql_quote(host)} AND ({likes}))")
    inside = "\n                      OR ".join(clauses)

    return f"""
        SELECT url, url_host_name AS host, url_path,
               content_mime_detected AS mime, stream, rn, n_stream
        FROM (
          SELECT url, url_host_name, url_path, content_mime_detected,
                 CASE WHEN {inside}
                      THEN 'in' ELSE 'out' END AS stream,
                 row_number() OVER (
                   PARTITION BY url_host_name,
                                CASE WHEN {inside}
                                     THEN 'in' ELSE 'out' END
                   ORDER BY xxhash64(to_utf8(concat(url, '{SEED}')))
                 ) AS rn,
                 count(*) OVER (
                   PARTITION BY url_host_name,
                                CASE WHEN {inside}
                                     THEN 'in' ELSE 'out' END
                 ) AS n_stream
          FROM ccindex
          WHERE crawl = '{SNAPSHOT}'
            AND subset = 'warc'
            AND fetch_status = 200
            AND content_mime_detected IN ({', '.join(HTML_MIMES)})
            AND content_languages = 'eng'
            AND url_host_name IN ({sql_in_list(sorted(by_host))})
        )
        WHERE (stream = 'in' AND rn <= {k_in * OVERFETCH})
           OR (stream = 'out' AND rn <= {k_out * OVERFETCH})
    """


def emit(rows, stream, domain):
    return [{
        "url": r["url"],
        "hint": f"bill_sample:{stream}",
        "host": r["host"],
        "registered_domain": domain,
        "url_path": r["url_path"],
        "mime": r["mime"],
        "stream": stream,
        "crawl": SNAPSHOT,
        "source": "openstates-bill-sections",
        "batch": BATCH_ID,
    } for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--in-per-site", type=int, default=14,
                    help="candidate pages per legislature, drawn from inside "
                         "its bill sections. Set so that even a section that "
                         "is mostly index pages still yields the 5 legal rows "
                         "leave-one-domain-out needs to score a publisher.")
    ap.add_argument("--out-per-site", type=int, default=2,
                    help="pages per legislature drawn from elsewhere on the "
                         "same site, outside its bill sections (default 2)")
    ap.add_argument("--output", default=OUTPUT_FILE)
    args = ap.parse_args()

    sections = [s for s in load_jsonl(SOURCES_FILE)
                if s["n_records"] >= MIN_RECORDS]
    all_sections = load_jsonl(SOURCES_FILE)
    print(f"Sections from {SOURCES_FILE}: {len(all_sections)}")
    print(f"  kept (>= {MIN_RECORDS} Open States records): {len(sections)} "
          f"across {len({s['host'] for s in sections})} sites")

    print(f"\nSnapshot: {SNAPSHOT}")
    print("Running two-stream sample query...")
    rows, _ = run_query(client(), build_sql(sections, args.in_per_site,
                                            args.out_per_site))

    by_host = collections.defaultdict(lambda: {"in": [], "out": []})
    depth = {}
    # A URL can be captured more than once in a single crawl, which puts two
    # index rows on one page. Keep the first so a page cannot be labeled twice.
    seen, n_dupe = set(), 0
    for r in rows:
        if r["url"] in seen:
            n_dupe += 1
            continue
        seen.add(r["url"])
        by_host[r["host"]][r["stream"]].append(r)
        if r["stream"] == "in":
            depth[r["host"]] = int(r["n_stream"])
    if n_dupe:
        print(f"  {n_dupe} repeat captures of the same URL, deduped")

    shortlist = sorted((h for h in by_host if depth.get(h, 0) >= MIN_ELIGIBLE),
                       key=lambda h: -depth[h])
    print(f"\nSites with captures inside their bill sections: {len(depth)}")
    print(f"  >= {MIN_ELIGIBLE} captured pages inside: {len(shortlist)}")
    if not shortlist:
        raise SystemExit("Empty shortlist - lower MIN_ELIGIBLE or recheck the sections file.")

    existing = load_existing_urls(EXISTING_BATCHES)
    print(f"Already-batched URLs (excluded): {len(existing)}")

    records, short, taken = [], [], set()
    for host in shortlist:
        domain = _extract(host).top_domain_under_public_suffix
        got = []
        for stream, quota in (("in", args.in_per_site), ("out", args.out_per_site)):
            fresh = [r for r in by_host[host][stream]
                     if r["url"] not in existing and r["url"] not in taken]
            fresh.sort(key=lambda r: rank(host, r["url"]))
            picked = fresh[:quota]
            got += emit(picked, stream, domain)
        for rec in got:
            taken.add(rec["url"])
        records += got
        if len(got) < args.in_per_site + args.out_per_site:
            short.append((host, len(got)))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    doms = {r["registered_domain"] for r in records}
    n_in = sum(1 for r in records if r["stream"] == "in")
    print(f"\n{len(records)} URLs across {len(shortlist)} sites "
          f"/ {len(doms)} registered domains")
    print(f"  {n_in} from inside bill sections, "
          f"{len(records) - n_in} from elsewhere on the same sites")
    if short:
        print(f"  {len(short)} site(s) came up short of quota: "
              f"{', '.join(f'{h}({n})' for h, n in short[:8])}")
    print(f"Written: {args.output}")
    print(f"\n{'inside':>8}  {'depth':>8}  site")
    for host in shortlist[:20]:
        got = sum(1 for r in records if r["host"] == host)
        print(f"{got:>8}  {depth[host]:>8,}  {host}")


if __name__ == "__main__":
    main()

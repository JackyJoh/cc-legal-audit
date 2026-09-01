"""
Pulls a fresh, unseen candidate pool for the two-tier deployment validation
(top-3 domains @ 0.85, everything else @ 0.70), scoped directly to the exact
hostnames known to carry legal text, rather than hoping they show up in a
generic random crawl sample. That's what raw_pool.jsonl/raw_pool_no_ca.jsonl
couldn't do: law.lis.virginia.gov never appeared in either 2.1M-URL random
pool, even though virginia.gov's own precision/recall on the held-out test
set is strong (81.2% / 70.3% @ 0.85) - the miss was a sampling gap, not a
model failure.

Two host lists, one query (one partition scan, so pulling both tiers here
costs the same as pulling one):

  top3_hosts  the exact hostnames legal-labeled training examples actually
              live on for cornell.edu, justice.gc.ca, and virginia.gov
              (pulled from data/processed/labeled_urls.jsonl directly, not
              guessed - virginia.gov alone has a dozen non-legal sibling
              hostnames sharing the same registered domain).
  minor_hosts every hostname in data/candidates/court_hostnames.jsonl (both
              its bare and www. spelling), the full CourtListener +
              legislature directory, not just the handful of domains that
              happened to get a confirmed legal label during that batch's
              thin per-host sample.

Uniform random sampling per host (seeded hash, not row order), capped at
PER_HOST_LIMIT rows each, so no single deep host (cornell.edu) crowds out
the query results before scoring even happens. The threshold/domain-cap
logic that turns this into an actual N=100-per-tier sample lives in
src/validation/sample_tiered_validation.py, not here - this script only
sources candidates.

Output: data/candidates/tier_validation_pool.jsonl, one {"url", "host",
"tier"} object per line.
"""
import json
import os
from collections import defaultdict

from dotenv import load_dotenv

from athena import client, run_query, sql_in_list

load_dotenv()

SNAPSHOT = "CC-MAIN-2026-12"
SEED = "44"
HOSTS_FILE = "data/candidates/court_hostnames.jsonl"
OUTPUT_FILE = "data/candidates/tier_validation_pool.jsonl"
PER_HOST_LIMIT = 300

TOP3_HOSTS = {
    "law.cornell.edu", "www.law.cornell.edu",
    "law.lis.virginia.gov", "www.law.lis.virginia.gov",
    "laws.justice.gc.ca", "www.laws.justice.gc.ca",
    "laws-lois.justice.gc.ca", "www.laws-lois.justice.gc.ca",
    "lois-laws.justice.gc.ca", "www.lois-laws.justice.gc.ca",
    "lois.justice.gc.ca", "www.lois.justice.gc.ca",
}

EXISTING_BATCHES = [
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
    "data/candidates/host_sample_batch.jsonl",
]


def load_minor_hosts(path):
    hosts = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            host = json.loads(line)["host"]
            bare = host[4:] if host.startswith("www.") else host
            hosts.add(bare)
            hosts.add("www." + bare)
    return hosts


def load_existing_urls(paths):
    urls = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    urls.add(json.loads(line)["url"])
    return urls


def build_sql(all_hosts):
    return f"""
        SELECT url, url_host_name AS host
        FROM (
          SELECT url, url_host_name,
                 row_number() OVER (
                   PARTITION BY url_host_name
                   ORDER BY xxhash64(to_utf8(concat(url, '{SEED}')))
                 ) AS rn
          FROM ccindex
          WHERE crawl = '{SNAPSHOT}'
            AND subset = 'warc'
            AND fetch_status = 200
            AND content_mime_detected IN ('text/html', 'application/xhtml+xml')
            AND content_languages = 'eng'
            AND url_host_name IN ({sql_in_list(all_hosts)})
        )
        WHERE rn <= {PER_HOST_LIMIT}
    """


def main():
    minor_hosts = load_minor_hosts(HOSTS_FILE)
    all_hosts = TOP3_HOSTS | minor_hosts
    print(f"top3 hostnames  : {len(TOP3_HOSTS)}")
    print(f"minor hostnames : {len(minor_hosts)}")
    print(f"combined        : {len(all_hosts)}")

    sql = build_sql(all_hosts)
    print(f"\nQuery length: {len(sql)} bytes (Athena limit 262144)")
    print(f"Snapshot: {SNAPSHOT}, per-host cap: {PER_HOST_LIMIT}")
    print("Running tier-candidate pull...")
    rows, stats = run_query(client(), sql)

    existing = load_existing_urls(EXISTING_BATCHES)
    print(f"Already-batched URLs (excluded): {len(existing)}")

    by_host = defaultdict(list)
    for row in rows:
        if row["url"] in existing:
            continue
        by_host[row["host"]].append(row["url"])

    # Athena doesn't guarantee row order without an explicit outer ORDER BY,
    # so without this sort the output file's line order (and everything
    # downstream that seeds off it) wouldn't reproduce across reruns even
    # though the per-host row_number() selection itself is deterministic.
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    n_written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for host in sorted(by_host):
            tier = "top3" if host in TOP3_HOSTS else "minor"
            for url in sorted(by_host[host]):
                f.write(json.dumps({"url": url, "host": host, "tier": tier}) + "\n")
                n_written += 1

    n_hosts_hit = len(by_host)
    n_top3_hit = len({h for h in by_host if h in TOP3_HOSTS})
    n_minor_hit = n_hosts_hit - n_top3_hit
    print(f"\nHosts queried, 0 captures: {len(all_hosts) - n_hosts_hit}")
    print(f"Hosts with captures      : {n_hosts_hit}  ({n_top3_hit} top3, {n_minor_hit} minor)")
    print(f"URLs written             : {n_written}  -> {OUTPUT_FILE}")
    print(f"Scanned {stats['scanned_gb']:.2f} GB, est ${stats['est_cost_usd']:.2f}")


if __name__ == "__main__":
    main()

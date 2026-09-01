"""
Pulls Common Crawl URLs for the CourtListener hostnames from
fetch_cl_hostnames.py, sampled per host, and writes the batch handed to the
labeling agent.

Legal positives are concentrated in a few registered domains, and
leave-one-domain-out recall on them is zero, so this step trades depth for
breadth: shallow samples from many publishers instead of more examples from
domains already covered.

Two draws per publisher: K_A random URLs (hard negatives, e.g. nav, listing,
and admin pages) and one URL per distinct two-segment path prefix up to K_B,
so a small section isn't drowned out by a high-fanout template like a
paginated bill-status page. Sampling uses a seeded hash for reproducibility.
Eligibility (HTML mime types, English) matches count_cl_captures.py.

Output: {"url", "hint"} per line, the shape intake.py and the labeling
prompt expect. hint names the stream, not the publisher, so the agent can't
tell it's looking at a known-legal domain.
"""
import hashlib
import json
import os

from dotenv import load_dotenv

from athena import client, run_query, sql_in_list

load_dotenv()

SNAPSHOT     = "CC-MAIN-2026-12"
SEED         = "42"
COUNTS_FILE  = "data/candidates/cc_host_counts.jsonl"
OUTPUT_FILE  = "data/candidates/host_sample_batch.jsonl"
BATCH_ID     = "host-sample-v1"

# Below this a publisher cannot support a clean sample. Set at the knee of the
# capture distribution: lowering it adds publishers that are almost all
# uscourts.gov siblings, which collapse into a single registered domain and so
# add labeling cost without adding a held-out group. See the threshold table
# printed by count_cl_captures.py.
MIN_ELIGIBLE = 200

K_A = 3    # uniform over URLs - hard negatives, kept small on purpose
K_B = 12   # uniform over path prefixes - section coverage

# Rows fetched per publisher per stream before dedupe against earlier batches,
# so publishers overlapping those batches still reach their k.
OVERFETCH = 3

# URLs already handed to the labeling agent. Excluded so nothing is relabeled.
EXISTING_BATCHES = [
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
]

# Both must match count_cl_captures.py. xhtml is HTML - matching text/html
# alone drops whole statute publishers.
HTML_MIMES = ("'text/html'", "'application/xhtml+xml'")
SQL_FAMILY = r"regexp_replace(url_host_name, '^www\.', '')"
# first two non-empty lowercased path segments; the query string is excluded
# because parameter permutations are exactly what inflates one template into
# tens of thousands of URLs
SQL_PREFIX = (r"array_join(slice(filter(split(lower(url_path), '/'), "
              r"x -> x <> ''), 1, 2), '/')")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_urls(paths):
    urls = set()
    for path in paths:
        if os.path.exists(path):
            urls.update(r["url"] for r in load_jsonl(path))
    return urls


def rank(*parts):
    """Deterministic order. blake2b, not the builtin hash() - string hashing
    is salted per process, so hash() would reshuffle on every run."""
    return hashlib.blake2b((SEED + "|".join(parts)).encode("utf-8"),
                           digest_size=8).digest()


def build_sql(hosts, fetch_a, fetch_b):
    """One scan, both streams.

    rn_url ranks every eligible capture within its publisher; rn_prefix ranks
    within publisher AND path prefix, so rn_prefix = 1 is one representative
    per section. Fetching both in a single pass avoids re-scanning the crawl
    partition twice.
    """
    return f"""
        SELECT url, url_host_name AS host, url_path,
               content_mime_detected AS mime,
               host_family, path_prefix, rn_url, rn_prefix
        FROM (
          SELECT url, url_host_name, url_path, content_mime_detected,
                 {SQL_FAMILY} AS host_family,
                 {SQL_PREFIX} AS path_prefix,
                 row_number() OVER (
                   PARTITION BY {SQL_FAMILY}
                   ORDER BY xxhash64(to_utf8(concat(url, '{SEED}')))
                 ) AS rn_url,
                 row_number() OVER (
                   PARTITION BY {SQL_FAMILY}, {SQL_PREFIX}
                   ORDER BY xxhash64(to_utf8(concat(url, '{SEED}')))
                 ) AS rn_prefix
          FROM ccindex
          WHERE crawl = '{SNAPSHOT}'
            AND subset = 'warc'
            AND fetch_status = 200
            AND content_mime_detected IN ({', '.join(HTML_MIMES)})
            AND content_languages = 'eng'
            AND url_host_name IN ({sql_in_list(hosts)})
        )
        WHERE rn_url <= {fetch_a} OR rn_prefix = 1
    """


def pick(rows, pub, existing, taken):
    """Stream A then Stream B for one publisher.

    A takes the lowest-ranked URLs outright. B then draws one row per prefix
    from prefixes A did not already land in, prefixes themselves shuffled by
    seeded hash rather than taken by size - taking the largest prefixes first
    would re-select the high-fanout template that Stream A already covers.
    """
    fresh = [r for r in rows if r["url"] not in existing and r["url"] not in taken]

    stream_a = sorted([r for r in fresh if int(r["rn_url"]) <= K_A * OVERFETCH],
                      key=lambda r: int(r["rn_url"]))[:K_A]
    a_urls = {r["url"] for r in stream_a}
    a_prefixes = {r["path_prefix"] for r in stream_a}

    by_prefix = {}
    for r in fresh:
        if r["url"] in a_urls or int(r["rn_prefix"]) != 1:
            continue
        if r["path_prefix"] in a_prefixes:
            continue
        by_prefix.setdefault(r["path_prefix"], r)

    order = sorted(by_prefix, key=lambda p: rank(pub["host"], p))
    stream_b = [by_prefix[p] for p in order[:K_B]]

    # If the host has fewer distinct prefixes than K_B there is nothing more
    # to cover; backfill from Stream A's pool so the publisher still reaches
    # its quota rather than silently coming up short.
    if len(stream_b) < K_B:
        picked = a_urls | {r["url"] for r in stream_b}
        spare = sorted([r for r in fresh if r["url"] not in picked],
                       key=lambda r: int(r["rn_url"]))
        stream_b += spare[:K_B - len(stream_b)]

    return stream_a, stream_b


def emit(rows, stream, pub):
    return [{
        "url": r["url"],
        "hint": f"host_sample:{stream}",
        "host": r["host"],
        "host_family": pub["host"],
        "registered_domain": pub["registered_domain"],
        "url_path": r["url_path"],
        "path_prefix": r["path_prefix"],
        "mime": r["mime"],
        "stream": stream,
        "crawl": SNAPSHOT,
        "source": "cl-court-domains",
    } for r in rows]


def main():
    counts = load_jsonl(COUNTS_FILE)
    shortlist = sorted((c for c in counts if c["n_eligible"] >= MIN_ELIGIBLE),
                       key=lambda c: -c["n_eligible"])
    print(f"Publishers measured      : {len(counts)}")
    print(f"  >= {MIN_ELIGIBLE} eligible captures: {len(shortlist)}")
    if not shortlist:
        raise SystemExit("Empty shortlist - lower MIN_ELIGIBLE or recheck the counts file.")

    print(f"Distinct registered doms : "
          f"{len({r['registered_domain'] for r in shortlist})}")
    print(f"Target                   : {len(shortlist)} x ({K_A} random + "
          f"{K_B} prefix) = {len(shortlist) * (K_A + K_B)} URLs")

    hosts = [h for r in shortlist for h in r["hosts"]]
    print(f"\nSnapshot: {SNAPSHOT}   {len(hosts)} hostnames")
    print("Running two-stream sample query...")
    rows, stats = run_query(client(), build_sql(hosts, K_A * OVERFETCH, K_B * OVERFETCH))

    by_family = {}
    for r in rows:
        by_family.setdefault(r["host_family"], []).append(r)

    existing = load_existing_urls(EXISTING_BATCHES)
    print(f"Already-batched URLs (excluded): {len(existing)}")

    records, short = [], []
    taken = set()
    for pub in shortlist:
        pool = by_family.get(pub["host"], [])
        stream_a, stream_b = pick(pool, pub, existing, taken)
        got = emit(stream_a, "random", pub) + emit(stream_b, "prefix", pub)
        for rec in got:
            rec["batch"] = BATCH_ID
            taken.add(rec["url"])
        records.extend(got)
        if len(got) < K_A + K_B:
            short.append((pub["host"], len(got)))

    # Interleave publishers. A labeling worker handed 15 consecutive URLs off
    # one domain is primed to label them the same way.
    records.sort(key=lambda r: rank(r["url"]))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n_a = sum(1 for r in records if r["stream"] == "random")
    n_b = len(records) - n_a
    prefixes = len({(r["host_family"], r["path_prefix"]) for r in records})
    print(f"\nPublishers represented   : "
          f"{len({r['host_family'] for r in records})}/{len(shortlist)}")
    print(f"Distinct path prefixes   : {prefixes}")
    if short:
        print(f"Publishers under quota   : {len(short)}")
        for host, n in short[:15]:
            print(f"    {host:<44} {n}")
    print(f"\nStream random / prefix   : {n_a} / {n_b}")
    print(f"Total URLs written       : {len(records)}  -> {OUTPUT_FILE}")
    print(f"Scanned {stats['scanned_gb']:.2f} GB, est ${stats['est_cost_usd']:.2f}")


if __name__ == "__main__":
    main()

"""
Draw the per-publisher URL sample out of the Common Crawl index.

This is the step that actually fixes the training set's shape. The positive
class is concentrated in a handful of registered domains, so char n-gram
TF-IDF can hit its held-out numbers by memorising one publisher's URL
templates. A per-publisher cap is what breaks that: no single templated
publisher gets to dominate the n-gram space, however many pages of it the
crawl holds. Run eval_grouped.py for the current concentration figures.

Sampling notes:

- The cap matters more than total volume. Raising k on a big publisher buys
  near-duplicate URL shapes; adding a publisher buys coverage.
- Whole hosts, no path filter. That deliberately pulls non-legal pages off
  legal domains - judge bios, careers pages, search forms, calendars. Those
  hard negatives are what the current training set lacks, and the likely
  cause of near-zero recall on the thin domains: the model has never seen a
  wa.gov URL that wasn't legal.
- Deterministic pseudorandom order, not rand(). Ordering by rand() gives a
  different sample every run, so nothing in the paper is reproducible;
  ordering by raw position clusters by crawl segment and path prefix, which
  reintroduces the template problem. Hashing url with a fixed seed is a
  seeded shuffle - the SQL analogue of the SEED = 42 the rest of the repo
  uses.
- Same eng + text/html + fetch_status 200 filters as the host-count step, so
  what gets labeled matches the distribution the classifier is scored on.

Two rules decide which hosts are samplable, both driven by what the host-count
step measured:

1. www-variant merge. Common Crawl reports 'example.gov' and
   'www.example.gov' as different hosts, and the directory only ever names one
   spelling. Treating them separately drops the captures held under the other
   one, and lets a publisher present under both spellings take two cap slots.
   Hosts are therefore folded into families on the leading 'www.', both in the
   SQL window and in the client-side cap.

2. Nothing else. The sampling frame is exactly the hostnames in the
   directory. Subdomains the crawl holds under the same registered domain are
   NOT pulled in: one registered domain covers wholly unrelated sites
   (revisor.mn.gov is statutes, gisdata.mn.gov is map data), and admitting
   them would mean deciding for ourselves which domains are legal enough to
   vouch for their subdomains - the hand-curation this project already
   rejected once.

Output: data/candidates/host_sample_batch.jsonl, {"url", "hint"} per line
(the shape intake.py and the labeling prompt expect) plus provenance fields
the labeling agent ignores. hint is "host_sample:<family>", triage-only, not
ground truth - same convention as build_label_batch.py.
"""
import hashlib
import json
import os

from dotenv import load_dotenv

from athena import client, run_query, sql_in_list

load_dotenv()

SNAPSHOT       = "CC-MAIN-2026-12"
SEED           = "42"
COUNTS_FILE    = "data/candidates/cc_host_counts.jsonl"
OUTPUT_FILE    = "data/candidates/host_sample_batch.jsonl"
BATCH_ID       = "host-sample-v1"

# URLs already handed to the labeling agent in an earlier batch. Excluded so
# nothing gets relabeled, same as fetch_targeted_urls.py.
EXISTING_BATCHES = [
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
]

# Shortlist parameters, chosen against the measured capture distribution.
MIN_ELIGIBLE = 200   # below this a publisher can't support a clean k-sample
MAX_HOSTS    = 100
K_PER_HOST   = 20
# fetched per family before dedupe against earlier batches, so publishers that
# overlap those batches still come out at K_PER_HOST
OVERFETCH    = 1.5

# folds www.example.gov and example.gov into one publisher, in SQL and here
SQL_FAMILY = r"regexp_replace(url_host_name, '^www\.', '')"


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


def main():
    # the counts file is already one row per publisher, www-variants folded,
    # and already restricted to the directory - no re-derivation needed here
    counts = load_jsonl(COUNTS_FILE)
    eligible = sorted((c for c in counts if c["n_eligible"] >= MIN_ELIGIBLE),
                      key=lambda c: -c["n_eligible"])
    shortlist = eligible[:MAX_HOSTS]

    print(f"Publishers measured        : {len(counts)}")
    print(f"  >= {MIN_ELIGIBLE} eligible captures  : {len(eligible)}")
    print(f"Shortlisted                : {len(shortlist)} (cap {MAX_HOSTS})")

    if not shortlist:
        raise SystemExit("Empty shortlist - lower MIN_ELIGIBLE or re-check the counts file.")

    by_source = {}
    for f in shortlist:
        for s in f["sources"]:
            by_source[s] = by_source.get(s, 0) + 1
    print(f"  by provenance            : {dict(sorted(by_source.items()))}")
    print(f"Distinct registered doms   : {len({f['registered_domain'] for f in shortlist})}")
    print(f"Target                     : {len(shortlist)} x {K_PER_HOST} = "
          f"{len(shortlist) * K_PER_HOST} URLs")

    # Query the exact hostnames the crawl reported for the shortlisted
    # families, rather than guessing www-variants that may not exist.
    hosts = [h for f in shortlist for h in f["hosts"]]
    fetch_k = int(K_PER_HOST * OVERFETCH)

    sql = f"""
        SELECT url, url_host_name AS host, host_family, url_path,
               content_mime_detected AS mime
        FROM (
          SELECT url, url_host_name, url_path, content_mime_detected,
                 {SQL_FAMILY} AS host_family,
                 row_number() OVER (
                   PARTITION BY {SQL_FAMILY}
                   ORDER BY xxhash64(to_utf8(concat(url, '{SEED}')))
                 ) AS rn
          FROM ccindex
          WHERE crawl = '{SNAPSHOT}'
            AND subset = 'warc'
            AND fetch_status = 200
            AND content_mime_detected = 'text/html'
            AND content_languages = 'eng'
            AND url_host_name IN ({sql_in_list(hosts)})
        )
        WHERE rn <= {fetch_k}
    """
    print(f"\nSnapshot: {SNAPSHOT}   {len(hosts)} hostnames, "
          f"fetching {fetch_k}/family before dedupe")
    print("Running per-publisher sample query...")

    rows, stats = run_query(client(), sql)

    existing = load_existing_urls(EXISTING_BATCHES)
    print(f"\nAlready-batched URLs (excluded): {len(existing)}")

    per_family = {}
    for row in rows:
        per_family.setdefault(row["host_family"], []).append(row)

    records = []
    short = []
    for fam in shortlist:
        got = [r for r in per_family.get(fam["host"], [])
               if r["url"] not in existing]
        kept = got[:K_PER_HOST]
        if len(kept) < K_PER_HOST:
            short.append((fam["host"], len(kept)))
        for r in kept:
            records.append({
                "url": r["url"],
                "hint": f"host_sample:{fam['host']}",
                "host": r["host"],
                "host_family": fam["host"],
                "registered_domain": fam["registered_domain"],
                "url_path": r["url_path"],
                "mime": r["mime"],
                "crawl": SNAPSHOT,
                "source": "cl-court-domains",
                "batch": BATCH_ID,
            })

    # Interleave across publishers rather than leaving the batch blocked by
    # host: a worker seeing 20 consecutive URLs off one domain is primed to
    # label them the same way. blake2b, not the builtin hash() - string
    # hashing is salted per process, so hash() would reshuffle every run.
    records.sort(key=lambda r: hashlib.blake2b(
        (SEED + r["url"]).encode("utf-8"), digest_size=8).digest())

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    covered = len({r["host_family"] for r in records})
    print(f"Publishers represented     : {covered}/{len(shortlist)}")
    if short:
        print(f"Publishers under k={K_PER_HOST}       : {len(short)}")
        for fam, n in short[:15]:
            print(f"    {fam:<44} {n}")
    print(f"\nTotal URLs written         : {len(records)}")
    print(f"Written                    : {OUTPUT_FILE}")
    print(f"Scanned {stats['scanned_gb']:.2f} GB, est ${stats['est_cost_usd']:.2f}")


if __name__ == "__main__":
    main()

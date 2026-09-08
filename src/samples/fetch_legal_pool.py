"""
Draws a uniform random sample of eligible pages from the legal domains, which
is the pool the precision sample gets taken from.

The classifier runs only inside court and legislature domains, so the number
worth measuring is precision on that surface rather than on the open web. Every
eligible capture across every host in legal_domains.jsonl has the same chance of
being drawn here, so the pool carries the same score distribution the corpus
will, and a sample of it estimates the precision the corpus will actually have.

Hosts are taken from legal_domains.jsonl under both their bare and www.
spellings, since Common Crawl stores those separately. Eligibility matches
count_domain_captures.py. Ordering by a seeded hash of the URL makes the draw
reproducible and independent of the order Athena happens to scan the partition
in, and taking a prefix of that order means a larger --n is a superset of a
smaller one, so raising it reuses every page already fetched.

Output: data/candidates/legal_pool.jsonl, one {"url", "host"} per line.
"""
import argparse
import json
import os

from dotenv import load_dotenv

from athena import client, run_query, sql_in_list

load_dotenv()

SNAPSHOT     = "CC-MAIN-2026-12"
DOMAINS_FILE = "data/candidates/legal_domains.jsonl"
OUTPUT_FILE  = "data/candidates/legal_pool.jsonl"
SEED         = "42"

HTML_MIMES = ("'text/html'", "'application/xhtml+xml'")
ELIGIBLE = ("fetch_status = 200 "
            f"AND content_mime_detected IN ({', '.join(HTML_MIMES)}) "
            "AND content_languages = 'eng'")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def spellings(host):
    bare = host[4:] if host.startswith("www.") else host
    return {bare, "www." + bare}


def build_sql(hosts, n):
    return f"""
        SELECT url, url_host_name AS host
        FROM ccindex
        WHERE crawl = '{SNAPSHOT}'
          AND subset = 'warc'
          AND {ELIGIBLE}
          AND url_host_name IN ({sql_in_list(hosts)})
        ORDER BY xxhash64(to_utf8(concat(url, '{SEED}')))
        LIMIT {n}
    """


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--n", type=int, default=8000,
                    help="pages to draw. Needs to be large enough that the "
                         "sparsest score stratum still holds its quota once "
                         "the pool is scored, which is why it is far larger "
                         "than the number of pages that get labeled.")
    ap.add_argument("--output", default=OUTPUT_FILE)
    args = ap.parse_args()

    domains = load_jsonl(DOMAINS_FILE)
    hosts = sorted({s for dom in domains for h in dom["hosts"] for s in spellings(h)})
    print(f"Legal domains            : {len(domains)}")
    print(f"Queried (both spellings) : {len(hosts)}")

    print(f"\nSnapshot: {SNAPSHOT}, drawing {args.n}")
    rows, stats = run_query(client(), build_sql(hosts, args.n))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"url": r["url"], "host": r["host"]}) + "\n")

    by_host = {}
    for r in rows:
        by_host[r["host"]] = by_host.get(r["host"], 0) + 1
    print(f"\nURLs written : {len(rows)}  -> {args.output}")
    print(f"Distinct hosts represented: {len(by_host)}")
    print(f"Scanned {stats['scanned_gb']:.2f} GB, est ${stats['est_cost_usd']:.2f}")


if __name__ == "__main__":
    main()

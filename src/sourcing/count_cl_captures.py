"""
Measures how many pages Common Crawl actually captured per candidate host,
so hosts can be ranked by real crawl depth rather than CourtListener docket
size (the two are uncorrelated, and a host with few captures can't support
a clean sample).

Two choices worth flagging:

1. Filters on url_host_name, never url_host_registered_domain, which is far
   coarser: mn.gov covers both revisor.mn.gov (statutes) and gisdata.mn.gov
   (map data), and af.mil covers one appeals court plus the entire Air
   Force. Each directory host is queried under both its bare and www.
   spellings, since Common Crawl stores those as separate hosts.

2. One query, one scan: the whole candidate set goes in a single IN list so
   the crawl partition is scanned once, not once per chunk.

n_pdf is collected only to document the HTML-only scope limit with a real
number, not because PDFs are sampled.

Output: data/candidates/cc_host_counts.jsonl
"""
import json
import os

from dotenv import load_dotenv

from athena import client, run_query, sql_in_list

load_dotenv()

SNAPSHOT    = "CC-MAIN-2026-12"
HOSTS_FILE  = "data/candidates/court_hostnames.jsonl"
OUTPUT_FILE = "data/candidates/cc_host_counts.jsonl"

# Eligibility mirrors fetch_candidate_urls.py's eng filter, plus HTML-only.
#
# HTML means the HTML family, not just 'text/html'. Many statute sites serve
# application/xhtml+xml; matching text/html alone drops law.lis.virginia.gov,
# one of the largest domains in the training set, to a single eligible page.
# XHTML is HTML, just a different serialization. PDFs are still excluded,
# which is what HTML-only is actually about.
HTML_MIMES = ("'text/html'", "'application/xhtml+xml'")
ELIGIBLE = ("fetch_status = 200 "
            f"AND content_mime_detected IN ({', '.join(HTML_MIMES)}) "
            "AND content_languages = 'eng'")


def load_hosts(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["host"]] = rec
    return records


def family_of(host):
    """The publisher, ignoring the www. prefix."""
    return host[4:] if host.startswith("www.") else host


def spellings(host):
    """Both spellings CC might have stored this host under."""
    bare = family_of(host)
    return {bare, "www." + bare}


def main():
    hosts = load_hosts(HOSTS_FILE)
    wanted = sorted({h for host in hosts for h in spellings(host)})
    print(f"Directory hostnames      : {len(hosts)}")
    print(f"Queried (both spellings) : {len(wanted)}")

    sql = f"""
        SELECT url_host_name AS host,
               url_host_registered_domain AS registered_domain,
               COUNT(*) AS n_all,
               COUNT_IF({ELIGIBLE}) AS n_eligible,
               COUNT_IF(content_mime_detected = 'application/pdf') AS n_pdf
        FROM ccindex
        WHERE crawl = '{SNAPSHOT}'
          AND subset = 'warc'
          AND url_host_name IN ({sql_in_list(wanted)})
        GROUP BY 1, 2
        ORDER BY n_eligible DESC
    """
    print(f"\nSnapshot: {SNAPSHOT}")
    print(f"Query length: {len(sql)} bytes (Athena limit 262144)")
    print("Running host-count query...")

    rows, stats = run_query(client(), sql)

    # One row per publisher: Common Crawl stores 'example.gov' and
    # 'www.example.gov' as separate hosts, but counting them apart would
    # split a publisher's depth in two and let it take two slots downstream.
    merged = {}
    for row in rows:
        host = row["host"]
        fam = family_of(host)
        directory_rec = next((hosts[h] for h in spellings(host) if h in hosts), None)
        rec = merged.setdefault(fam, {
            "host": fam,
            "hosts": [],
            "registered_domain": row["registered_domain"],
            "n_all": 0,
            "n_eligible": 0,
            "n_pdf": 0,
            "sources": directory_rec["sources"] if directory_rec else [],
            "n_courts": directory_rec["n_courts"] if directory_rec else 0,
            "example_court": directory_rec["example_court"] if directory_rec else None,
        })
        rec["hosts"].append(host)
        rec["n_all"] += int(row["n_all"])
        rec["n_eligible"] += int(row["n_eligible"])
        rec["n_pdf"] += int(row["n_pdf"])

    records = sorted(merged.values(), key=lambda r: -r["n_eligible"])
    for rec in records:
        rec["hosts"].sort()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    seen = {r["host"] for r in records}
    missing = sorted(h for h in hosts if family_of(h) not in seen)
    total_elig = sum(r["n_eligible"] for r in records)
    total_pdf = sum(r["n_pdf"] for r in records)
    total_all = sum(r["n_all"] for r in records)

    print(f"\nPublishers found in crawl: {len(records)}")
    print(f"Directory hosts, 0 caps  : {len(missing)}")
    print(f"  (of {len(hosts)} directory hostnames)")
    print(f"Captures  all/eligible   : {total_all} / {total_elig}")
    if total_all:
        print(f"PDF share of captures    : {total_pdf}/{total_all} = {total_pdf / total_all:.2%}"
              "   (excluded, HTML-only scope)")

    for cut in (500, 200, 100, 50, 20):
        n = sum(1 for r in records if r["n_eligible"] >= cut)
        print(f"  publishers with >= {cut:>4} eligible captures: {n}")

    print(f"\n--- publishers by eligible captures ---")
    print(f"{'publisher':<40} {'eligible':>9} {'pdf':>7} {'spellings':>9}  sources")
    for rec in records:
        src = ",".join(rec["sources"]) or "-"
        print(f"{rec['host']:<40} {rec['n_eligible']:>9} {rec['n_pdf']:>7} "
              f"{len(rec['hosts']):>9}  {src}")

    print(f"\nWritten: {OUTPUT_FILE}")
    print(f"Scanned {stats['scanned_gb']:.2f} GB, est ${stats['est_cost_usd']:.2f}")


if __name__ == "__main__":
    main()

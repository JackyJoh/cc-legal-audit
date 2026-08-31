"""
How deeply does Common Crawl actually capture each candidate host?

Ranks by CC capture depth, not by CourtListener opinion volume - the two are
uncorrelated, and a court with a huge CL docket but a handful of pages in the
crawl cannot support a clean sample and is not worth a labeling slot.

Two deliberate choices:

1. Filter on url_host_name, never url_host_registered_domain. A registered
   domain is far coarser than a hostname: mn.gov covers revisor.mn.gov
   (statutes) and gisdata.mn.gov (map data alike), and af.mil covers one
   military appeals court plus the entire Air Force. Filtering on the
   registered domain therefore measures thousands of sites the directory
   never named and that will never be sampled. Each directory host is queried
   under both its bare and www. spellings, since Common Crawl treats those as
   separate hosts and the directory only records one of them.

2. One query, one scan. The whole candidate set goes in a single IN list
   rather than being chunked - every chunk would re-scan the same crawl
   partition and multiply the bill. Conditional aggregation gets the
   all/eligible/pdf breakdowns out of that one pass.

n_pdf is collected purely to document the HTML-only scope limitation with a
real number in the paper, not because PDFs are sampled.

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

# Eligibility = the deployment distribution the classifier actually sees.
# Mirrors fetch_candidate_urls.py's eng filter, plus the HTML-only decision.
#
# HTML means the HTML family, not the single string 'text/html'. Many statute
# sites are served as application/xhtml+xml, and matching text/html exactly
# discards them wholesale - law.lis.virginia.gov, one of the largest domains
# in the existing training set, drops to a single eligible page. XHTML is
# HTML; the distinction here is serialization, not content type. PDFs are
# still excluded, which is what the HTML-only decision was actually about.
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

    # One row per publisher, not per hostname. Common Crawl stores
    # 'example.gov' and 'www.example.gov' as separate hosts, but they are the
    # same site: counting them apart splits a publisher's depth in two and
    # would let it take two slots downstream. hosts[] keeps the spellings the
    # crawl actually holds, which is what the sampling query needs to ask for.
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

"""
Measures how many pages Common Crawl actually captured per legal domain, so
domains can be ranked by real crawl depth instead of by CourtListener docket
size or Open States bill count, neither of which correlates with it.

Two choices worth flagging:

1. Queries url_host_name, never url_host_registered_domain. A registered
   domain is far coarser than the thing being measured: ohio.gov covers both
   www.legislature.ohio.gov and every unrelated state agency subdomain,
   af.mil covers one appeals court plus the entire Air Force. Querying by
   url_host_name and summing back up to each domain's own known host list
   (from legal_domains.jsonl) keeps the count to exactly the hosts that will
   actually be sampled from, whether that domain is a single court host or a
   legislature's several known hosts. Each host is also queried under both
   its bare and www. spellings, since Common Crawl stores those separately.

2. One query, one scan: every candidate host across every domain goes into a
   single IN list, so the crawl partition is scanned once, not once per
   domain.

n_pdf is collected only to document the HTML-only scope limit with a real
number, not because PDFs are sampled.

Output: data/candidates/cc_domain_counts.jsonl
"""
import json
import os
import sys
from collections import defaultdict

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "common"))
from athena import client, run_query, sql_in_list  # noqa: E402

load_dotenv()

SNAPSHOT     = "CC-MAIN-2026-12"
DOMAINS_FILE = "data/candidates/legal_domains.jsonl"
OUTPUT_FILE  = "data/candidates/cc_domain_counts.jsonl"

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


def load_domains(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def family_of(host):
    """The publisher, ignoring the www. prefix."""
    return host[4:] if host.startswith("www.") else host


def spellings(host):
    """Both spellings CC might have stored this host under."""
    bare = family_of(host)
    return {bare, "www." + bare}


def main():
    domains = load_domains(DOMAINS_FILE)

    # every host spelling maps back to the domain_key that owns it - a court
    # row owns exactly one host, a legislature row can own several
    family_to_domain = {}
    for dom in domains:
        for host in dom["hosts"]:
            family_to_domain[family_of(host)] = dom["domain_key"]

    wanted = sorted({s for fam in family_to_domain for s in spellings(fam)})
    print(f"Legal domains             : {len(domains)}")
    print(f"Distinct hosts (families) : {len(family_to_domain)}")
    print(f"Queried (both spellings)  : {len(wanted)}")

    sql = f"""
        SELECT url_host_name AS host,
               COUNT(*) AS n_all,
               COUNT_IF({ELIGIBLE}) AS n_eligible,
               COUNT_IF(content_mime_detected = 'application/pdf') AS n_pdf
        FROM ccindex
        WHERE crawl = '{SNAPSHOT}'
          AND subset = 'warc'
          AND url_host_name IN ({sql_in_list(wanted)})
        GROUP BY 1
        ORDER BY n_eligible DESC
    """
    print(f"\nSnapshot: {SNAPSHOT}")
    print(f"Query length: {len(sql)} bytes (Athena limit 262144)")
    print("Running domain-count query...")

    rows, stats = run_query(client(), sql)

    # Roll both spellings up to the host family, then every family up to the
    # domain it belongs to. A legislature domain can carry several genuinely
    # distinct hosts, not just spelling variants of one, so this is a second
    # aggregation step on top of the spelling merge, not the same one.
    by_domain = defaultdict(lambda: {"hosts_found": set(), "n_all": 0,
                                      "n_eligible": 0, "n_pdf": 0})
    for row in rows:
        dom_key = family_to_domain.get(family_of(row["host"]))
        if dom_key is None:
            continue
        agg = by_domain[dom_key]
        agg["hosts_found"].add(row["host"])
        agg["n_all"] += int(row["n_all"])
        agg["n_eligible"] += int(row["n_eligible"])
        agg["n_pdf"] += int(row["n_pdf"])

    domains_by_key = {d["domain_key"]: d for d in domains}
    records = []
    for dom_key, agg in by_domain.items():
        dom = domains_by_key[dom_key]
        records.append({
            "domain_key": dom_key,
            "unit": dom["unit"],
            "type": dom["type"],
            "hosts": dom["hosts"],
            "hosts_found_in_cc": sorted(agg["hosts_found"]),
            "n_all": agg["n_all"],
            "n_eligible": agg["n_eligible"],
            "n_pdf": agg["n_pdf"],
        })
    records.sort(key=lambda r: -r["n_eligible"])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    zero_cap = [d["domain_key"] for d in domains if d["domain_key"] not in by_domain]
    total_elig = sum(r["n_eligible"] for r in records)
    total_pdf = sum(r["n_pdf"] for r in records)
    total_all = sum(r["n_all"] for r in records)

    print(f"\nDomains found in crawl : {len(records)}")
    print(f"Domains, 0 captures    : {len(zero_cap)} (of {len(domains)})")
    print(f"Captures  all/eligible : {total_all} / {total_elig}")
    if total_all:
        print(f"PDF share of captures  : {total_pdf}/{total_all} = {total_pdf / total_all:.2%}"
              "   (excluded, HTML-only scope)")

    for cut in (10000, 1000, 500, 200, 100, 50, 20):
        n = sum(1 for r in records if r["n_eligible"] >= cut)
        print(f"  domains with >= {cut:>5} eligible captures: {n}")

    print(f"\n--- domains by eligible captures ---")
    print(f"{'domain_key':<32} {'type':<12} {'eligible':>9} {'pdf':>7} {'hosts':>6}")
    for rec in records[:40]:
        print(f"{rec['domain_key']:<32} {rec['type']:<12} {rec['n_eligible']:>9} "
              f"{rec['n_pdf']:>7} {len(rec['hosts_found_in_cc']):>6}")
    if len(records) > 40:
        print(f"  ... {len(records) - 40} more, see {OUTPUT_FILE}")

    if zero_cap:
        print(f"\ndomains with 0 eligible captures: {zero_cap}")

    print(f"\nWritten: {OUTPUT_FILE}")
    print(f"Scanned {stats['scanned_gb']:.2f} GB, est ${stats['est_cost_usd']:.2f}")


if __name__ == "__main__":
    main()

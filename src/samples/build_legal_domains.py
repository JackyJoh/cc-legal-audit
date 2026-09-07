"""
Merges the court and legislature host lists into one list of legal domains,
each carrying the sampling unit that's correct for it.

Courts and legislatures need different units. A state legislature is its own
registrant (one apex domain per state), so two hostnames under the same
registered domain are the same publisher and should merge - Ohio's
codes.ohio.gov and www.legislature.ohio.gov both resolve to ohio.gov, and
that's one domain, not two. Federal courts are the opposite: almost all of
them share one registrant, uscourts.gov, handed out to each court as a
subdomain, so collapsing by registered domain would merge 86 separate courts
into a single entry. Courts stay at hostname granularity; legislatures (the
seed list plus every host Open States found bill text on) merge at the
registered-domain level.

Reads:
  data/candidates/court_hostnames.jsonl          courts + legislature seed hosts
  data/candidates/register_sources_bills.jsonl   Open States bill-path sources

Writes:
  data/candidates/legal_domains.jsonl, one row per domain unit. Court rows are
  keyed by host and carry n_courts/jurisdictions/example_court. Legislature
  rows are keyed by registered domain and carry every known host for that
  domain plus any bill-path patterns Open States found under it.
"""
import json
import os
from collections import defaultdict

from fetch_cl_hostnames import registered_domain

COURT_HOSTNAMES_FILE = "data/candidates/court_hostnames.jsonl"
BILL_SOURCES_FILE = "data/candidates/register_sources_bills.jsonl"
OUTPUT_FILE = "data/candidates/legal_domains.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_court_rows(court_hostnames):
    """One row per host. uscourts.gov alone would merge 86 distinct courts
    into a single registered domain, so courts don't collapse."""
    rows = []
    for rec in court_hostnames:
        if "cl-courts" not in rec["sources"]:
            continue
        rows.append({
            "domain_key": rec["host"],
            "unit": "host",
            "type": "court",
            "hosts": [rec["host"]],
            "registered_domain": rec["registered_domain"],
            "n_courts": rec["n_courts"],
            "jurisdictions": rec["jurisdictions"],
            "example_court": rec["example_court"],
            "bill_patterns": [],
        })
    return rows


def build_legislature_rows(court_hostnames, bill_sources):
    """One row per registered domain. A state's legislature seed host and
    whatever host Open States found bill text on are almost always the same
    organization reached through different subdomains, so these merge."""
    seed_hosts = {rec["host"] for rec in court_hostnames
                  if "seed:us-legislature" in rec["sources"]}
    bill_hosts = {rec["host"] for rec in bill_sources}

    by_domain = defaultdict(lambda: {"hosts": set(), "bill_patterns": []})
    for host in seed_hosts | bill_hosts:
        by_domain[registered_domain(host)]["hosts"].add(host)
    for rec in bill_sources:
        by_domain[registered_domain(rec["host"])]["bill_patterns"].append({
            "host": rec["host"],
            "pattern": rec["pattern"],
            "n_records": rec["n_records"],
            "examples": rec["examples"],
        })

    rows = []
    for dom, data in sorted(by_domain.items()):
        rows.append({
            "domain_key": dom,
            "unit": "registered_domain",
            "type": "legislature",
            "hosts": sorted(data["hosts"]),
            "registered_domain": dom,
            "n_courts": None,
            "jurisdictions": [],
            "example_court": None,
            "bill_patterns": sorted(data["bill_patterns"],
                                     key=lambda p: -p["n_records"]),
        })
    return rows


def main():
    court_hostnames = load_jsonl(COURT_HOSTNAMES_FILE)
    bill_sources = load_jsonl(BILL_SOURCES_FILE)

    court_rows = build_court_rows(court_hostnames)
    legislature_rows = build_legislature_rows(court_hostnames, bill_sources)
    rows = court_rows + legislature_rows

    n_multi_host = sum(1 for r in legislature_rows if len(r["hosts"]) > 1)
    n_with_bills = sum(1 for r in legislature_rows if r["bill_patterns"])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"courts       : {len(court_rows)} domains (host-level)")
    print(f"legislatures : {len(legislature_rows)} domains (registered-domain-level)")
    print(f"  multi-host : {n_multi_host} domains backed by 2+ hosts")
    print(f"  with bills : {n_with_bills} domains have an Open States bill pattern")
    print(f"total        : {len(rows)} legal domains")
    print(f"written      : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

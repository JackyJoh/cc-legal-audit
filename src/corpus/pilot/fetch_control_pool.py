"""
Builds the two page sets for the MinHash pilot.

  legal:   Laya v2 keeps (>= 0.85) from fetch_legal_pages.py's pool, trimmed
           to each site's target so the mix matches open-web legal
  control: non-legal pages, one random site per legal site, matched on crawl
           depth and pages drawn. Matched by site because near-duplicates come
           from shared site templates; a uniform draw would have almost none.

Sites are hosts minus "www.". Legal sites and hosts on legal_domains.jsonl are
excluded from the control.

Writes to data/pilot/: legal_set.jsonl, control_pool.jsonl (with WARC
pointers), control_sites.jsonl (the site matching).

Then: python src/common/fetch_warc_text.py --input data/pilot/control_pool.jsonl
          --output data/pilot/control_text.jsonl
"""
import hashlib
import math
import os
import sys
from collections import Counter, defaultdict

from dotenv import load_dotenv

from sampling import (ELIGIBLE, OUT_DIR, SEED, SITE_SQL, SNAPSHOT, load_jsonl,
                      one_capture_per_url, pages_sql, site_of, spellings, write_jsonl)
from athena import client, run_query, sql_in_list  # noqa: E402  (path set by sampling)

load_dotenv()

LAYA_MODEL   = os.path.normpath("models/laya-legal")
LAYA_MAX_LEN = 1024
LAYA_CUT     = 0.85

LEGAL_POOL    = f"{OUT_DIR}/legal_pool.jsonl"
LEGAL_TEXT    = f"{OUT_DIR}/legal_pool_text.jsonl"
LAYA_SCORES   = f"{OUT_DIR}/laya_legal_scores.jsonl"
DOMAINS_FILE  = "data/candidates/legal_domains.jsonl"

LEGAL_SET     = f"{OUT_DIR}/legal_set.jsonl"
CONTROL_POOL  = f"{OUT_DIR}/control_pool.jsonl"
CONTROL_SITES = f"{OUT_DIR}/control_sites.jsonl"


def bucket_of(n):
    return int(math.floor(math.log2(n)))


def seeded_order(url):
    return hashlib.sha1((url + SEED).encode("utf-8")).hexdigest()


def build_legal_set():
    """Laya keeps with text, at most each site's target, in seeded order."""
    p_legal = {r["url"]: float(r["p_legal"]) for r in load_jsonl(LAYA_SCORES)
               if os.path.normpath(str(r.get("model"))) == LAYA_MODEL
               and r.get("max_len") == LAYA_MAX_LEN}
    if not p_legal:
        sys.exit(f"no {LAYA_MODEL} scores at max_len {LAYA_MAX_LEN} in {LAYA_SCORES}; "
                 "run score_laya.py first (see fetch_legal_pages.py)")
    target = {r["site"]: r["target"] for r in load_jsonl(LEGAL_POOL)}

    kept = defaultdict(list)
    for r in load_jsonl(LEGAL_TEXT):
        if r.get("text") and p_legal.get(r["url"], 0.0) >= LAYA_CUT:
            kept[site_of(r["url"])].append({"url": r["url"], "site": site_of(r["url"]),
                                            "text": r["text"]})
    legal = []
    for site, rows in kept.items():
        rows.sort(key=lambda r: seeded_order(r["url"]))
        legal += rows[:target[site]]

    short = {s: (len(kept.get(s, [])), t) for s, t in target.items() if len(kept.get(s, [])) < t}
    return sorted(legal, key=lambda r: r["url"]), short


def excluded_membership():
    """Hosts and registered domains the control may not draw from."""
    hosts, regs = set(), set()
    for dom in load_jsonl(DOMAINS_FILE):
        for host in dom["hosts"]:
            hosts |= spellings(host[4:] if host.startswith("www.") else host)
        if dom["unit"] == "registered_domain":
            regs.add(dom["registered_domain"])
    return hosts, regs


def depth_sql(legal_sites, ex_hosts, ex_regs):
    """Page count per legal site, plus random control candidates per depth bucket."""
    return f"""
        WITH depth AS (
            SELECT {SITE_SQL} AS site,
                   count(*) AS n,
                   bool_or(url_host_name IN ({sql_in_list(ex_hosts)})
                           OR url_host_registered_domain IN ({sql_in_list(ex_regs)})) AS excluded
            FROM ccindex
            WHERE crawl = '{SNAPSHOT}'
              AND subset = 'warc'
              AND {ELIGIBLE}
            GROUP BY 1
        ),
        ranked AS (
            SELECT site, n,
                   row_number() OVER (
                       PARTITION BY CAST(floor(log2(n)) AS integer)
                       ORDER BY xxhash64(to_utf8(concat(site, '{SEED}')))) AS rn
            FROM depth
            WHERE NOT excluded
        )
        SELECT site, n, 'legal' AS role, 0 AS rn FROM depth
        WHERE site IN ({sql_in_list(legal_sites)})
        UNION ALL
        SELECT site, n, 'control' AS role, rn FROM ranked
        WHERE rn <= {len(legal_sites)}
    """


def match_sites(legal_quota, depth_rows):
    """Pair each legal site with the next unused control site in its bucket."""
    legal_depth = {r["site"]: int(r["n"]) for r in depth_rows if r["role"] == "legal"}
    candidates = defaultdict(list)
    for r in sorted((r for r in depth_rows if r["role"] == "control"), key=lambda r: int(r["rn"])):
        candidates[bucket_of(int(r["n"]))].append((r["site"], int(r["n"])))

    matches, unmatched = [], []
    used = Counter()
    for site in sorted(legal_quota):
        if site not in legal_depth:
            unmatched.append((site, "not found in the index"))
            continue
        b = bucket_of(legal_depth[site])
        if used[b] >= len(candidates[b]):
            unmatched.append((site, f"no control site left in bucket {b}"))
            continue
        c_site, c_n = candidates[b][used[b]]
        used[b] += 1
        matches.append({"legal_site": site, "legal_depth": legal_depth[site],
                        "bucket": b, "control_site": c_site, "control_depth": c_n,
                        "quota": min(legal_quota[site], c_n)})
    return matches, unmatched


def main():
    print("1. Legal set")
    legal, short = build_legal_set()
    legal_quota = Counter(r["site"] for r in legal)
    write_jsonl(LEGAL_SET, legal)
    print(f"  {len(legal)} pages, {len(legal_quota)} sites  -> {LEGAL_SET}")
    for site, (got, want) in sorted(short.items()):
        print(f"  SHORT {site}: {got} Laya keeps, target {want}")

    print("\n2. Site depth and control candidates")
    ex_hosts, ex_regs = excluded_membership()
    ex_hosts |= {h for s in legal_quota for h in spellings(s)}
    athena = client()
    depth_rows, _ = run_query(athena, depth_sql(sorted(legal_quota), ex_hosts, ex_regs))

    matches, unmatched = match_sites(legal_quota, depth_rows)
    write_jsonl(CONTROL_SITES, matches)
    print(f"  {len(matches)} sites matched  -> {CONTROL_SITES}")
    for site, why in unmatched:
        print(f"  UNMATCHED {site}: {why} ({legal_quota[site]} legal pages lose their control)")

    print("\n3. Control pages")
    quota = {m["control_site"]: m["quota"] for m in matches}
    matched_to = {m["control_site"]: m["legal_site"] for m in matches}
    page_rows, _ = run_query(athena, pages_sql(quota))
    control = [{**r, "matched_to": matched_to[r["site"]]} for r in one_capture_per_url(page_rows)]
    write_jsonl(CONTROL_POOL, control)
    print(f"  {len(control)} pages  -> {CONTROL_POOL}")

    print(f"\n  {'legal site':<32}{'depth':>9}{'pages':>7}   {'control site':<32}{'depth':>9}{'pages':>7}")
    got = Counter(r["site"] for r in control)
    for m in sorted(matches, key=lambda m: -legal_quota[m["legal_site"]]):
        print(f"  {m['legal_site'][:31]:<32}{m['legal_depth']:>9,}{legal_quota[m['legal_site']]:>7}   "
              f"{m['control_site'][:31]:<32}{m['control_depth']:>9,}{got[m['control_site']]:>7}")

    print(f"\nNext: python src/common/fetch_warc_text.py --input {CONTROL_POOL} "
          f"--output {OUT_DIR}/control_text.jsonl")


if __name__ == "__main__":
    main()

"""
Pulls court hostnames from the CourtListener courts API to build a list of
domains that are near-certainly legal (each one is a court's own website),
plus a hardcoded list of all 50 state legislature sites (CourtListener
indexes case law only, so no legislature ever appears in it). Writes that
host list for fetch_cl_urls.py to sample Common Crawl URLs from.

CourtListener supplies domain coverage only, never URL strings: pulling
URLs directly from CL would shift the distribution the classifier sees at
inference (CL's own storage paths and PDF-heavy layout barely appear in
Common Crawl), and char n-gram TF-IDF would learn "courtlistener" itself as
a legal-class feature.

The CL side samples a strided subset of pages by default (SAMPLE_PAGES),
not the full court list, since the API's hourly request cap makes a full
pull take hours; set SAMPLE_PAGES = None for a complete pull.

Output: data/candidates/court_hostnames.jsonl, one record per hostname
(host, registered_domain, sources, n_courts, jurisdictions, example_court).
"""
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from urllib.parse import urlparse

import tldextract
from dotenv import load_dotenv

load_dotenv()

CL_API      = "https://www.courtlistener.com/api/rest/v4/courts/"
USER_AGENT  = "cc-legal-audit/research (University of Florida)"
OUTPUT_FILE = "data/candidates/court_hostnames.jsonl"

# CourtListener enforces two limits: a per-minute burst limit and a per-hour cap.
#  MIN_INTERVAL satisfies the burst limit but says nothing about the hourly
# one, so a long enough run walks into the hourly wall and then sits in a very
# long retry-after. Keep the whole job under HOURLY_BUDGET requests and the 
# burst spacing is all the pacing needed.
MIN_INTERVAL   = 13.0
HOURLY_BUDGET  = 50
# above this, a 429 is the hourly wall rather than the burst limit
MAX_RETRY_WAIT = 120.0

# Written immediately per page so an interrupted run keeps what it already
# paid for, given how scarce the hourly quota is.
CACHE_FILE = "data/candidates/_cl_courts_cache.jsonl"

# v4 caps page_size regardless of what is requested; this is the cap, not a
# preference. It is what makes the full court list cost so many requests.
PAGE_SIZE = 20

# Pages are strided, not taken off the front; the endpoint is ordered by
# judicial hierarchy (SCOTUS -> circuits -> districts -> state systems), so a
# prefix would be the federal judiciary with states cut off, not a sample.
# Doubling this value reuses every cached page, since the strided plan for
# 2N contains the plan for N. Set to None for a full pull.
SAMPLE_PAGES = 28

# Closed enumeration: all 50 US states have exactly one official legislature
# site, not a curated list, so there's no recall gap by construction.
# Purposely choose to not extend this list to statute sites or other
# non-US legal domains/sources beacuse they have no 'limit' or way to create
# an exhaustive list. Therefore a list beyond these is simlply what I have thought
# of, and not a reproduceable list.
SEED_HOSTS = {
    "seed:us-legislature": [
        "legislature.state.al.us", "www.akleg.gov", "www.azleg.gov",
        "www.arkleg.state.ar.us", "leginfo.legislature.ca.gov",
        "leg.colorado.gov", "www.cga.ct.gov", "legis.delaware.gov",
        "www.leg.state.fl.us", "www.legis.ga.gov", "www.capitol.hawaii.gov",
        "legislature.idaho.gov", "www.ilga.gov", "iga.in.gov",
        "www.legis.iowa.gov", "www.kslegislature.gov", "legislature.ky.gov",
        "www.legis.la.gov", "legislature.maine.gov", "mgaleg.maryland.gov",
        "malegislature.gov", "www.legislature.mi.gov", "www.revisor.mn.gov",
        "billstatus.ls.state.ms.us", "revisor.mo.gov", "leg.mt.gov",
        "nebraskalegislature.gov", "www.leg.state.nv.us",
        "www.gencourt.state.nh.us", "www.njleg.state.nj.us", "www.nmlegis.gov",
        "www.nysenate.gov", "www.ncleg.gov", "www.ndlegis.gov",
        "codes.ohio.gov", "www.oklegislature.gov", "www.oregonlegislature.gov",
        "www.legis.state.pa.us", "webserver.rilegislature.gov",
        "www.scstatehouse.gov", "sdlegislature.gov", "www.capitol.tn.gov",
        "statutes.capitol.texas.gov", "le.utah.gov",
        "legislature.vermont.gov", "law.lis.virginia.gov", "app.leg.wa.gov",
        "www.wvlegislature.gov", "docs.legis.wisconsin.gov", "www.wyoleg.gov",
    ],
}

# pinned local suffix list, no PSL fetch at runtime
_extract = tldextract.TLDExtract(suffix_list_urls=())


def registered_domain(host):
    return _extract(host).top_domain_under_public_suffix


def get_json(url, token):
    """One GET, honouring the server's retry-after on throttle."""
    req = urllib.request.Request(url, headers={
        "Authorization": f"Token {token}",
        "User-Agent": USER_AGENT,
    })
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            wait = float(exc.headers.get("retry-after", MIN_INTERVAL)) + 1
            # A short wait is the burst limit clearing. A long one is the
            # hourly wall, and sleeping through that silently looks identical
            # to a hang. Everything fetched so far is already cached, so stop
            # and let the next run resume for free.
            if wait > MAX_RETRY_WAIT:
                mins = wait / 60
                raise SystemExit(
                    f"\nHourly quota exhausted ({HOURLY_BUDGET}/hour). "
                    f"Quota returns in ~{mins:.0f} min.\n"
                    f"Pages fetched so far are cached in {CACHE_FILE}. "
                    f"Re-run then and it resumes without re-spending them."
                )
            print(f"  throttled, waiting {wait:.0f}s")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 4:
                raise
            print(f"  retry in {2 ** attempt}s after {exc}")
            time.sleep(2 ** attempt)


def page_plan(total_pages, sample_pages):
    """Which page numbers to fetch, spread evenly across the whole range."""
    if sample_pages is None or sample_pages >= total_pages:
        return list(range(1, total_pages + 1)), "full pull"
    stride = total_pages / sample_pages
    pages = sorted({min(total_pages, int(i * stride) + 1)
                    for i in range(sample_pages)})
    return pages, f"strided sample, {len(pages)}/{total_pages} pages, every ~{stride:.0f}"


def load_cache(path):
    """Pages already paid for on an earlier run: {page_number: payload}."""
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cache[rec["page"]] = rec
    return cache


def append_cache(path, page, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "page": page,
            "count": payload["count"],
            "results": payload["results"],
        }) + "\n")


def fetch_courts(token, sample_pages=SAMPLE_PAGES):
    """Yield raw court records from the CL courts endpoint.

    Page 1 establishes the total, then a strided page plan covers the rest.
    Every page is cached to disk the moment it arrives, and cached pages are
    served without spending a request, so an interrupted run resumes free.
    """
    base = f"{CL_API}?page_size={PAGE_SIZE}"
    cache = load_cache(CACHE_FILE)
    if cache:
        print(f"  cache: {len(cache)} page(s) already fetched -> {sorted(cache)}")

    if 1 in cache:
        first = cache[1]
    else:
        first = get_json(base, token)
        append_cache(CACHE_FILE, 1, first)
        cache[1] = first

    count = first["count"]
    total_pages = max(1, -(-count // PAGE_SIZE))
    pages, mode = page_plan(total_pages, sample_pages)
    to_fetch = [p for p in pages if p not in cache]

    print(f"  {count} courts over {total_pages} pages of {PAGE_SIZE} -> {mode}")
    print(f"  pages: {pages}")
    print(f"  need {len(to_fetch)} request(s), {len(pages) - len(to_fetch)} from cache")
    if len(to_fetch) > HOURLY_BUDGET:
        print(f"  WARNING: {len(to_fetch)} requests exceeds the {HOURLY_BUDGET}/hour "
              f"budget; run will stall on the hourly wall")
    est = max(0, len(to_fetch) - 1) * MIN_INTERVAL / 60
    print(f"  estimated ~{est:.0f} min at {MIN_INTERVAL:.0f}s/request\n")

    fetched = 0
    for i, page in enumerate(pages):
        if page in cache:
            payload = cache[page]
            src = "cached"
        else:
            if fetched:
                time.sleep(MIN_INTERVAL)
            payload = get_json(f"{base}&page={page}", token)
            append_cache(CACHE_FILE, page, payload)
            fetched += 1
            src = "fetched"
        print(f"  [{i + 1:>2}/{len(pages)}] page {page:>3}/{total_pages}  "
              f"+{len(payload['results'])} courts ({src})", flush=True)
        yield from payload["results"]


def host_of(raw_url):
    """Hostname out of a court's own website URL, or None if unusable."""
    if not raw_url:
        return None
    if "://" not in raw_url:
        raw_url = "http://" + raw_url
    host = (urlparse(raw_url).hostname or "").strip().lower().rstrip(".")
    # a bare TLD or an IP literal is not a usable sampling target
    if not host or "." not in host or host.replace(".", "").isdigit():
        return None
    return host


def main():
    token = os.environ["CL_API_TOKEN"]

    hosts = {}  # host -> accumulated provenance

    def add(host, source, court=None):
        rec = hosts.setdefault(host, {
            "host": host,
            "registered_domain": registered_domain(host),
            "sources": [],
            "n_courts": 0,
            "jurisdictions": [],
            "example_court": None,
        })
        if source not in rec["sources"]:
            rec["sources"].append(source)
        if court:
            rec["n_courts"] += 1
            j = court.get("jurisdiction")
            if j and j not in rec["jurisdictions"]:
                rec["jurisdictions"].append(j)
            if rec["example_court"] is None:
                rec["example_court"] = court.get("full_name")

    print("--- CourtListener courts endpoint ---")
    n_total = n_in_use = n_no_host = 0
    for court in fetch_courts(token):
        n_total += 1
        if not court.get("in_use"):
            continue
        n_in_use += 1
        host = host_of(court.get("url"))
        if host is None:
            n_no_host += 1
            continue
        add(host, "cl-courts", court)

    print(f"\ncourts returned      : {n_total}")
    print(f"  in_use=true        : {n_in_use}")
    print(f"  unusable url field : {n_no_host}")
    print(f"distinct CL hostnames: {len(hosts)}")

    # Coverage check on the stride: the endpoint is ordered by judicial
    # hierarchy, so seeing several jurisdiction codes here is the evidence
    # that the sample spans it rather than stopping inside the federal block.
    seen_j = Counter(j for r in hosts.values() for j in r["jurisdictions"])
    print(f"jurisdictions covered: {len(seen_j)}  {dict(sorted(seen_j.items()))}")

    print("\n--- hand-added seed hosts ---")
    for source, seed_hosts in SEED_HOSTS.items():
        before = len(hosts)
        for host in seed_hosts:
            add(host, source)
        print(f"  {source:<24} {len(seed_hosts):>3} listed, {len(hosts) - before:>3} new")

    records = sorted(hosts.values(), key=lambda r: (r["registered_domain"], r["host"]))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n_reg = len({r["registered_domain"] for r in records})
    print(f"\nhostnames          : {len(records)}")
    print(f"registered domains : {n_reg}")
    print(f"Written            : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

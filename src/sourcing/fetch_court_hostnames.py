"""
Build the legal-publisher hostname directory that the crawl-index steps
sample from.

CourtListener is used here as a *directory of court domains*, not as a source
of URL strings. Pulling URLs straight from CL would shift the distribution the
classifier sees at inference (CL's own storage paths and PDF-heavy record
layout are largely absent from Common Crawl), and char 3-5 gram TF-IDF would
happily learn 'courtlistener' and 'pdf' as legal-class features. So CL supplies
domain coverage; Common Crawl supplies the actual URL strings.

CL indexes case law, so no legislature appears in it. SEED_HOSTS below closes
exactly that gap and nothing else - see the note there for why it stops at the
state legislatures rather than growing into a curated list of legal
publishers.

The CL side is a strided page sample by default (SAMPLE_PAGES), not every
court. The API's hourly request cap and its small maximum page size together
make a full pull cost hours of waiting on quota. Hostnames also repeat heavily
across courts - many courts share one state judiciary domain - so a spread
sample recovers most of the distinct-host coverage for a fraction of it. The
run prints the page plan and its request count before spending anything; set
SAMPLE_PAGES = None for the complete pull.

Output: data/candidates/court_hostnames.jsonl, one record per hostname:
  {"host": ..., "registered_domain": ..., "sources": [...], "n_courts": N,
   "jurisdictions": [...], "example_court": "..."}

registered_domain is computed with tldextract, which uses the same Public
Suffix List that Common Crawl's url_host_registered_domain column is built
from - verified against domains already present in the labeled set
(wa.gov, justice.gc.ca, virginia.gov).
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

# CourtListener enforces two limits: a per-minute burst limit and, the one
# that actually binds, a per-hour cap. MIN_INTERVAL satisfies the burst limit
# but says nothing about the hourly one, so a long enough run walks into the
# hourly wall and then sits in a very long retry-after. Keep the whole job
# under HOURLY_BUDGET requests and the burst spacing is all the pacing needed.
MIN_INTERVAL   = 13.0
HOURLY_BUDGET  = 50
# above this, a 429 is the hourly wall rather than the burst limit
MAX_RETRY_WAIT = 120.0

# Every page fetched is written here immediately and reused on the next run.
# Quota this scarce must not be spent twice on the same page, and an
# interrupted run has to keep what it already paid for - both earlier runs
# lost every page they fetched because results were only written at the end.
CACHE_FILE = "data/candidates/_cl_courts_cache.jsonl"

# v4 caps page_size regardless of what is requested; this is the cap, not a
# preference. It is what makes the full court list cost so many requests.
PAGE_SIZE = 20

# Number of pages to actually fetch. Set to None for a full pull.
#
# Pages are strided across the whole range, not taken off the front. The
# endpoint is ordered by the court's `position` field, which is the judicial
# hierarchy - SCOTUS, then circuits, then district courts, then state systems.
# So a prefix is not a sample of US courts, it is the federal judiciary with
# the state courts cut off. Striding costs the same number of requests and
# covers the whole hierarchy.
#
# Doubling this reuses every page already cached rather than re-spending on
# them, because the strided plan for 2N contains the plan for N - the new
# pages interleave between the old ones, so coverage stays even across the
# hierarchy and only the additions cost quota.
SAMPLE_PAGES = 28

# The one gap CourtListener structurally cannot fill: it indexes case law, so
# no legislature ever appears in it.
#
# This is a complete enumeration of a closed population - there are exactly 50
# US states and each has one official legislature site - which is why it is
# defensible where an open-ended curated list would not be. It can be stated
# in the methods section in one sentence and has no recall gap by
# construction: nothing was selected, the population was enumerated.
#
# Deliberately NOT extended to federal statute sites or non-US sources. Those
# have no closure rule, so any such list encodes which publishers the author
# happened to think of, and an omission in it is invisible and unauditable.
# That is the same objection that got the rule-based whitelist classifier
# dropped, and it applies to a sampling frame as much as to a classifier.
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
                    f"Pages fetched so far are cached in {CACHE_FILE} - "
                    f"re-run then and it resumes without re-spending them."
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

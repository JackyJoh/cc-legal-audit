"""
Finds out where each state legislature publishes its bills.

Bills sit in a specific part of each legislature's website: one state keeps
them under /Bills/, another under /li/*/measures/. Nobody has written down
which state uses which, and picking them by hand is the kind of human
judgment this project already threw out once. So we ask an outside authority
that already knows.

Open States records, for every state bill, a link back to that bill's page on
its own legislature's site. This script reads a sample of those records,
throws the individual addresses away, and keeps only the shape: which
website, and which part of it. The result is a map of where to look, which
the next stage uses to pull real candidate URLs out of Common Crawl.

What it does NOT do: download documents, touch Common Crawl, or decide which
publishers are worth using. It only reports what the authority says.

How it works:

  1. Ask every one of the 50 states separately, since Open States rejects a
     query carrying no state filter. All 50 are covered rather than a subset,
     because which states are worth using gets decided later from real Common
     Crawl depth, and excluding any here would be a guess made before that is
     measured.
  2. For each bill, take the address it lives at and reduce it to a pattern:
     drop the filename, and replace any part of the path containing a digit
     with a wildcard, since those are dates, session names and bill numbers.
     So
       /li/b2025_26/measures/hb2001/  becomes  /li/%/measures/%
     That wildcard form is what a Common Crawl query can match on.
  3. Group by website and pattern, and write one row each with a count of how
     many records pointed there. High counts are well-attested; a count of 1
     is probably a one-off worth ignoring.

Addresses are read for their location, not their file format. A PDF sits in
the same section of a site as the HTML around it, so it identifies that
section just as well. Whether PDFs are worth collecting as documents is a
question for the Common Crawl query, not for this step.

Every page is cached as it arrives, so an interrupted or rate-limited run
keeps what it already paid for and the next run resumes free.

Output: data/candidates/register_sources_<source>.jsonl
"""
import argparse
import collections
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv()

USER_AGENT = "cc-legal-audit/research (University of Florida)"
OUTPUT_FILE = "data/candidates/register_sources_{source}.jsonl"
CACHE_FILE = "data/candidates/_{source}_pages_cache.jsonl"

# Open States publishes no rate limit, so there is no interval to respect in
# advance and the 429 handler below is the whole brake: it waits exactly as
# long as the server asks. Measured behaviour is roughly 15 requests, then a
# 14 second cooldown. This value is only the fallback when a 429 arrives
# without a Retry-After header.
MIN_INTERVAL = 13.0
# above this, a Retry-After is an hourly wall rather than a burst limit, and
# sleeping through it looks identical to a hang
MAX_RETRY_WAIT = 120.0

# Every state has to be asked separately: Open States refuses an unfiltered
# /bills query ("either 'jurisdiction' or 'q' required"). All 50 states have
# exactly one legislature, so this is a closed enumeration rather than a list
# someone chose, and it cannot have a recall gap by construction.
STATES = ["ak", "al", "ar", "az", "ca", "co", "ct", "de", "fl", "ga", "hi",
          "ia", "id", "il", "in", "ks", "ky", "la", "ma", "md", "me", "mi",
          "mn", "mo", "ms", "mt", "nc", "nd", "ne", "nh", "nj", "nm", "nv",
          "ny", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut",
          "va", "vt", "wa", "wi", "wv", "wy"]

SOURCES = {
    "bills": {
        "url": "https://v3.openstates.org/bills",
        # include=sources is required: without it the response carries no
        # addresses at all and this script would find nothing
        "params": {"per_page": 20, "include": "sources", "sort": "updated_desc"},
        "auth": ("X-Api-Key", "{key}", "OPEN_STATES_API_KEY"),
        "per_page": 20,
        # no deliberate pause: Open States answers 429 with a Retry-After
        # rather than banning, so the handler below is the whole rate limit.
        # Raise this if it spends more time backing off than fetching.
        "interval": 0.0,
        # per state, so 3 x 50 = 150 requests for up to 3000 bills
        "default_pages": 3,
        # one request per state rather than per page, see STATES above
        "jurisdictions": STATES,
        "url_field": lambda rec: [s.get("url") for s in (rec.get("sources") or [])],
    },
}


def get_json(url, headers):
    """One GET, honouring the server's Retry-After on throttle."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **headers})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                # the body carries the server's reason; without it a 400 is
                # indistinguishable from any other malformed request
                body = exc.read()[:500].decode("utf-8", "replace")
                raise SystemExit(f"  HTTP {exc.code} from {url}\n  {body}")
            wait = float(exc.headers.get("retry-after", MIN_INTERVAL)) + 1
            if wait > MAX_RETRY_WAIT:
                raise SystemExit(
                    f"  rate limit wall: server asks for {wait / 60:.0f} min. "
                    f"Everything so far is cached, rerun later to resume free.")
            print(f"  throttled, waiting {wait:.0f}s")
            time.sleep(wait)
    raise SystemExit("  gave up after 5 attempts")


def build_url(base, params, page=None):
    if page is not None:
        params = {**params, "page": page}
    return f"{base}?{urllib.parse.urlencode(params)}"


def records_of(payload):
    for key in ("results", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def section_pattern(path):
    """Reduce a path to the shape a Common Crawl query can match.

    Drops the filename, then replaces any segment containing a digit with a
    wildcard, since those are dates, session names and document ids. Returns
    None for a path with nothing left to match on.
    """
    segments = [s for s in path.split("/") if s]
    if len(segments) < 2:
        return None
    kept = ["%" if any(c.isdigit() for c in s) else s for s in segments[:-1]]
    if all(s == "%" for s in kept):
        return None
    return "/" + "/".join(kept) + "/%"


def load_cache(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["unit"]: json.loads(line)["payload"]
                for line in f if line.strip()}


def append_cache(path, unit, payload):
    """One line per request, written the moment it lands, so a run that hits
    a rate wall keeps everything it already paid for."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"unit": unit, "payload": payload}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--source", choices=sorted(SOURCES), default="bills")
    ap.add_argument("--pages", type=int, default=None,
                    help="pages per state, asked of all 50 states "
                         "(default 3, so 150 requests for up to 3000 bills)")
    args = ap.parse_args()

    spec = SOURCES[args.source]
    if args.pages is None:
        args.pages = spec["default_pages"]
    interval = spec.get("interval", MIN_INTERVAL)
    header_name, header_fmt, env_var = spec["auth"]
    key = os.environ.get(env_var)
    if not key:
        raise SystemExit(f"missing {env_var} in .env")
    headers = {header_name: header_fmt.format(key=key)}

    cache_path = CACHE_FILE.format(source=args.source)
    cache = load_cache(cache_path)
    if cache:
        print(f"resuming: {len(cache)} pages already cached")

    print(f"--- {args.source} ---")
    # Every state is asked separately, since the API refuses a query with no
    # state filter. All 50 are covered rather than a subset: which states are
    # worth using is decided later from real Common Crawl depth, so excluding
    # any here would be a guess made before that is measured.
    states = spec["jurisdictions"]
    plan = [(f"{state}:{page}",
             build_url(spec["url"],
                       {**spec["params"], "jurisdiction": state}, page))
            for state in states for page in range(1, args.pages + 1)]
    print(f"  {len(states)} states x {args.pages} page(s) = {len(plan)} requests")
    todo = [p for p in plan if p[0] not in cache]
    print(f"  {len(todo)} to fetch, {len(plan) - len(todo)} cached, "
          f"~{len(todo) * interval / 60:.1f} min")

    for i, (label, url) in enumerate(plan, 1):
        if label in cache:
            continue
        print(f"  {label} ({i}/{len(plan)})")
        payload = get_json(url, headers)
        append_cache(cache_path, label, payload)
        cache[label] = payload
        time.sleep(interval)
    labels = [label for label, _ in plan]

    groups = collections.defaultdict(lambda: {"n": 0, "examples": []})
    n_records = n_pdf = n_nourl = 0
    for label in labels:
        for rec in records_of(cache.get(label, {})):
            n_records += 1
            urls = [u for u in spec["url_field"](rec) if u]
            if not urls:
                n_nourl += 1
                continue
            for u in urls:
                parsed = urllib.parse.urlparse(u)
                # counted, not excluded: a PDF sits in the same section as the
                # HTML around it, so it locates that section just as well
                if parsed.path.lower().endswith(".pdf"):
                    n_pdf += 1
                pattern = section_pattern(parsed.path)
                if not parsed.netloc or not pattern:
                    continue
                g = groups[(parsed.netloc, pattern)]
                g["n"] += 1
                if len(g["examples"]) < 3:
                    g["examples"].append(u)

    out_path = OUTPUT_FILE.format(source=args.source)
    rows = [{"host": h, "pattern": p, "n_records": g["n"], "examples": g["examples"]}
            for (h, p), g in groups.items()]
    rows.sort(key=lambda r: -r["n_records"])
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n  {n_records} records, {n_nourl} with no address, "
          f"{n_pdf} pointing at a pdf")
    print(f"  {len(rows)} website+section pairs across "
          f"{len({r['host'] for r in rows})} websites")
    print(f"  written: {out_path}\n")
    print(f"{'records':>8}  {'website':<34} section")
    for r in rows[:20]:
        print(f"{r['n_records']:>8}  {r['host']:<34} {r['pattern']}")


if __name__ == "__main__":
    main()

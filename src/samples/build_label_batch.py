"""
Turns the raw Athena pool (fetch_candidate_urls.py output) into the actual
batch that is handed to the LLM labeling agent.

Legal URLs are a tiny fraction of the raw crawl (about 0.2-0.3%, per the
archived rule-based classifier's own validation numbers), so pure random
sampling starves the labeling agent of legal candidates. We fix that by
mixing three buckets into one batch:

  1. prefilter hits: every URL in the raw pool that the archived
     rule-based classifier (WL or KW match) flagged as plausibly legal.
     This boosts legal density, but the labeling agent still has to
     confirm or reject each one, since the rule-based classifier isn't
     ground truth.
  2. raw random: a fixed-size random sample of URLs the classifier did NOT
     flag, so the non-legal side isn't limited to "things that looked
     legal-ish and got rejected." The regression still needs to see the
     ordinary diversity of the general web. Sized independently of the
     prefilter bucket, not matched to it, since the two buckets are
     covering different targets (legal recall vs. non-legal diversity).
  3. homepage negatives: one bare-domain URL per entry in the archived
     wl_candidates.txt whitelist. Random sampling almost never lands on a
     domain's bare homepage, but "right domain, wrong page type" (an
     index/menu page instead of an actual filing) is exactly the failure
     mode the project's README calls out, so these are added on purpose.

Output: data/candidates/candidates.jsonl, one {"url": ..., "hint": ...}
object per line. `hint` is one of "prefilter_wl" / "prefilter_kw" /
"raw_random" / "homepage" - a triage aid for the labeling agent, not
ground truth.
"""
import json
import os
import random
import sys
from urllib.parse import urlparse

ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "archive", "rule-based")
sys.path.insert(0, ARCHIVE_DIR)
from URL_Classifier import classify, InWhitelist  # noqa: E402

# config
SEED             = 42
RAW_POOL_FILE    = "data/candidates/raw_pool.jsonl"
WL_FILE          = os.path.join(ARCHIVE_DIR, "wl_candidates.txt")
OUTPUT_FILE      = "data/candidates/candidates.jsonl"
TARGET_PREFILTER = 1500   # cap on the prefilter-hit bucket
TARGET_RAW       = 1500   # size of the raw random bucket, independent of prefilter


def load_raw_pool(path):
    urls = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            urls.append(json.loads(line)['url'])
    return urls


def load_homepage_domains(path):
    domains = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            domains.append(parts[-1])
    return domains


def main():
    random.seed(SEED)

    urls = load_raw_pool(RAW_POOL_FILE)
    print(f"Raw pool          : {len(urls)} URLs")

    wl_hits, kw_hits, misses = [], [], []
    for url in urls:
        if not classify(url):
            misses.append(url)
            continue
        host = urlparse(url).netloc.split(':', 1)[0]
        (wl_hits if InWhitelist(host) else kw_hits).append(url)

    prefilter_hits = wl_hits + kw_hits
    print(f"Prefilter hits    : {len(prefilter_hits)}  ({len(wl_hits)} WL, {len(kw_hits)} KW)")

    if len(prefilter_hits) > TARGET_PREFILTER:
        prefilter_hits = random.sample(prefilter_hits, TARGET_PREFILTER)

    n_raw = min(TARGET_RAW, len(misses))
    raw_random = random.sample(misses, n_raw) if n_raw else []

    homepage_domains = load_homepage_domains(WL_FILE)
    homepage_urls = [f"https://{d}/" for d in homepage_domains]

    # prefilter_hits may be a random.sample() mix of wl_hits and kw_hits after
    # the cap above, so re-check InWhitelist per-url instead of slicing the
    # original lists.
    batch = (
        [{"url": u, "hint": "prefilter_wl" if InWhitelist(urlparse(u).netloc.split(':', 1)[0]) else "prefilter_kw"}
         for u in prefilter_hits] +
        [{"url": u, "hint": "raw_random"} for u in raw_random] +
        [{"url": u, "hint": "homepage"} for u in homepage_urls]
    )
    random.shuffle(batch)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for row in batch:
            f.write(json.dumps(row) + "\n")

    print(f"Raw random        : {len(raw_random)}")
    print(f"Homepage negatives: {len(homepage_urls)}")
    print(f"Batch total       : {len(batch)}")
    print(f"Written           : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

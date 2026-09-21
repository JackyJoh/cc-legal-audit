"""
Measures Jev's precision on the open web the only affordable way: score a
uniform crawl draw, keep what Jev calls legal, and hand-label just that.

Why only the kept pages need labels. Precision is legal-kept over kept, so
the denominator is the kept set and nothing else has to be looked at. The
number that can't be measured this way is the false-positive rate, whose
denominator is every non-legal page in the draw; at a 0.12% base rate the
FPR a usable open-web filter needs is around 0.01%, which takes tens of
thousands of labeled negatives to see even one miss. Precision sidesteps
that: 100k pages scored yields ~60 legal kept at a 0.90 cut plus whatever
leaks, and labeling that handful gives precision with a real interval.
If every kept page comes back legal, the Wilson lower bound at n kept is
what can be claimed; the summary prints it so the run says whether it was
big enough.

This is fetch_flagged_urls.py with Jev in place of the TF-IDF model, and it
reuses that script's dedupe and file conventions. Two deliberate
differences. The draw keeps each pool row's WARC pointer columns instead of
stripping them to a bare URL, so locating 100k pages costs nothing rather
than a chunked Athena lookup. And fetching and scoring overlap: each page
goes onto a queue the moment its text lands, a packer fills Jev requests
off that queue by token budget, and a second pool sends them. Scoring is a
tenth the wall-clock of fetching, so the gain isn't speed; it's that kept
pages and their scores show up minutes in, and a run killed halfway has
scored everything it fetched.

No flags. It is one specific experiment; the numbers that define it are the
constants below. Every stage resumes: the pool is pulled once, page text is
cached as it lands, Jev scores are appended per request, and a rerun picks
up wherever the last one stopped.

Stages:
  1. Pool.   data/candidates/raw_pool.jsonl, ~600k uniform URLs with
             pointers. Pulled via fetch_candidate_urls if missing (one
             Athena scan, about $0.50).
  2. Draw.   First N_DRAW of a seeded shuffle, same seed as the flagged run.
  3. Fetch + score, overlapped. 6 fetch workers (WARC range-GETs +
             trafilatura), appended to the cache per page; 6 Jev workers
             fed by a packer that flushes on a full budget or an idle queue.
  4. Keep.   p_legal >= CUT, minus anything already labeled or batched.
  5. Write.  The batch (URLs only) and the scores, in separate files, so the
             labeling agent never sees what Jev thinks.
  6. Compact the text cache: text is kept only for the pages that were kept.

Outputs:
  data/labels/jev_openweb_scores.jsonl            every scored page, p_legal, tokens
  data/candidates/jev_openweb_batch.jsonl         kept URLs for labeling, no score
  data/candidates/jev_openweb_kept_scores.jsonl   p_legal per kept URL

Then label the batch with prompts/legal_url_labeling_task.md and read the
result with eval_is_legal.py --truth <run file> --pred <kept scores>.

Requires JEV_API_KEY and the AWS/Athena keys in .env.
"""
import json
import math
import os
import queue
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "samples"))
sys.path.insert(0, HERE)
from fetch_warc_text import (N_WORKERS, build_row, load_jsonl,  # noqa: E402
                             needs_retry, resolve_pointers)
from fetch_flagged_urls import EXISTING, load_existing_urls  # noqa: E402
import fetch_candidate_urls as pool_puller  # noqa: E402
import label_is_legal as jev  # noqa: E402

N_DRAW = 100_000
CUT    = 0.90
SEED   = 42   # same as fetch_flagged_urls, so a draw from the same pool is the same draw

# How long the packer lets a part-filled request sit before sending it
# anyway. Fetch workers land pages at a steady clip, so this only fires at
# the tail of the run or if the crawl is throttling.
IDLE_FLUSH_S = 5.0

POOL_FILE     = "data/candidates/raw_pool.jsonl"
TEXT_CACHE    = "data/candidates/_pool_text_cache.jsonl"
POINTER_CACHE = "data/candidates/_pool_pointers_cache.jsonl"
JEV_SCORES    = "data/labels/jev_openweb_scores.jsonl"
JEV_SKIPPED   = "data/labels/jev_openweb_scores.skipped.jsonl"
OUTPUT_FILE   = "data/candidates/jev_openweb_batch.jsonl"
SCORES_FILE   = "data/candidates/jev_openweb_kept_scores.jsonl"
BATCH_ID      = "jev-openweb-v1"
PRICE_PER_M   = 0.042   # USD per million input tokens, jev-latest

# Beyond fetch_flagged_urls' list: the batches that exist now but didn't
# when that list was written.
ALREADY_LABELED = EXISTING + [
    "data/candidates/flagged_sample_batch.jsonl",
    "data/candidates/legal_sample_batch.jsonl",
    "data/validation/precision_sets.jsonl",
]


def ensure_pool():
    if os.path.exists(POOL_FILE):
        return
    print(f"{POOL_FILE} missing; pulling a fresh uniform pool "
          f"(one Athena scan, ~$0.50)")
    athena = pool_puller.client()
    total = pool_puller.population_size(athena)
    percent = min(100.0, pool_puller.DEFAULT_N / total * 100)
    print(f"  population {total:,}, BERNOULLI({percent:.6f}) for ~{pool_puller.DEFAULT_N:,}")
    rows = pool_puller.sample(athena, percent)
    os.makedirs(os.path.dirname(POOL_FILE) or ".", exist_ok=True)
    with open(POOL_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {len(rows):,} rows")


def draw(pool_file, n):
    """fetch_flagged_urls.draw, but returning the whole pool row so the
    pointer columns ride along and locating the pages is free."""
    by_url = {}
    for r in load_jsonl(pool_file):
        by_url[r["url"]] = r
    ordered = sorted(by_url)
    random.Random(SEED).shuffle(ordered)
    print(f"pool: {len(ordered):,} unique URLs, drawing {n:,}")
    return [by_url[u] for u in ordered[:n]]


def to_doc(row):
    return {"url": row["url"], "text": row["text"][:jev.DEFAULT_MAX_CHARS]}


def fetch_and_score(rows, pointers, cache_path, labeler, already_scored):
    """Stage 3. Fetch every drawn page not already cached and push each
    page with text to the labeler as it lands; pages already cached but
    not yet scored go first.

    The cache handling is fetch_flagged_urls.fetch_text's: append per page,
    claim a URL on disk before touching it so a page that crashes the
    extractor is skipped next run instead of crashing it again, last line
    wins on read. Text is not held in memory: it goes to the cache file and
    to the queue and that is all, so the draw can be as large as the disk.

    Returns {url: status row without text} for every drawn URL.
    """
    have = {}
    if os.path.exists(cache_path):
        for r in load_jsonl(cache_path):
            have[r["url"]] = r
        print(f"  cached: {len(have):,} pages")

    q = queue.Queue()
    n_queued = [0]

    def enqueue(row):
        if row.get("text") and row["url"] not in already_scored:
            q.put(to_doc(row))
            n_queued[0] += 1

    def scorer():
        """Packer thread: fill by token budget, flush on a full request or
        an idle queue, stop on the sentinel."""
        packer = jev.Packer(jev.DEFAULT_STATE_TOKEN_BUDGET)
        while True:
            try:
                doc = q.get(timeout=IDLE_FLUSH_S)
            except queue.Empty:
                b = packer.flush()
                if b:
                    labeler.submit(b)
                continue
            if doc is None:
                b = packer.flush()
                if b:
                    labeler.submit(b)
                return
            b = packer.add(doc)
            if b:
                labeler.submit(b)

    t = threading.Thread(target=scorer, name="jev-packer", daemon=True)
    t.start()

    drawn = {r["url"] for r in rows}
    for url, r in have.items():
        if url in drawn:
            enqueue(r)
    if n_queued[0]:
        print(f"  {n_queued[0]:,} cached pages queued for scoring")

    todo = [(r, pointers.get(r["url"])) for r in rows
            if r["url"] not in have or needs_retry(have[r["url"]])]
    print(f"  {len(todo):,} to fetch")
    if todo:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        lock = threading.Lock()
        done = 0
        with open(cache_path, "a", encoding="utf-8") as cache:

            def guarded(item):
                with lock:
                    cache.write(json.dumps(
                        {"url": item[0]["url"], "text": None,
                         "skip_reason": "no result, extraction did not return"}) + "\n")
                    cache.flush()
                return build_row(item)

            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                futures = [pool.submit(guarded, item) for item in todo]
                for f in as_completed(futures):
                    row = f.result()
                    with lock:
                        cache.write(json.dumps(row, ensure_ascii=False) + "\n")
                        done += 1
                        if done % 500 == 0:
                            cache.flush()
                            print(f"  fetched {done:,}/{len(todo):,}   "
                                  f"scored {labeler.stats['done']:,}  "
                                  f"kept so far {labeler.stats['yes']}")
                    enqueue(row)
                    have[row["url"]] = row

    q.put(None)
    t.join()

    for r in have.values():
        r.pop("text", None)
    return have


def wilson_lower(k, n, z=1.96):
    """Lower 95% bound on a proportion with k of n; the number you can claim."""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d


def compact(cache_path, keep_text, p_by_url):
    """Rewrite the cache line by line, dropping page text for every page
    that was not kept. The url, skip reason and Jev score stay, which is
    all a rerun needs to know the page is done. Later lines win, as when
    the cache is read, so the rewrite also removes retried duplicates."""
    before = os.path.getsize(cache_path)
    last = {}
    for r in load_jsonl(cache_path):
        last[r["url"]] = r
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for url in sorted(last):
            r = last[url]
            if url not in keep_text:
                r = {"url": url, "text": None,
                     "skip_reason": r.get("skip_reason") or "not kept, text dropped"}
            if url in p_by_url:
                r["p_legal"] = p_by_url[url]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, cache_path)
    print(f"compacted {cache_path}: {before / 1e6:.0f} MB -> "
          f"{os.path.getsize(cache_path) / 1e6:.1f} MB, text kept for {len(keep_text)} pages")


def main():
    print("1. Pool")
    ensure_pool()
    rows = draw(POOL_FILE, N_DRAW)

    print("\n2. WARC pointers")
    pointers = resolve_pointers(rows, POINTER_CACHE)

    print("\n3. Fetch + score")
    already = {r["url"] for r in load_jsonl(JEV_SCORES)} if os.path.exists(JEV_SCORES) else set()
    print(f"  {len(already):,} pages already scored")
    os.makedirs(os.path.dirname(JEV_SCORES) or ".", exist_ok=True)
    with jev.make_client() as client:
        labeler = jev.Labeler(client, JEV_SCORES, JEV_SKIPPED)
        have = fetch_and_score(rows, pointers, TEXT_CACHE, labeler, already)
        labeler.close()
    with_text = sum(1 for r in rows if have.get(r["url"], {}).get("text_chars"))

    scored = {r["url"]: r for r in load_jsonl(JEV_SCORES)}
    drawn_scored = [scored[r["url"]] for r in rows if r["url"] in scored]
    tokens = {}
    for r in drawn_scored:
        tokens[r["request"]] = r["input_tokens"]
    total_tokens = sum(tokens.values())

    print("\n4. Keep")
    kept = [r for r in drawn_scored if r["p_legal"] >= CUT]
    print(f"  {len(kept)} of {len(drawn_scored):,} scored pages at or above {CUT} "
          f"({len(kept) / max(1, len(drawn_scored)):.4%})")
    existing = load_existing_urls(ALREADY_LABELED)
    fresh = sorted((r for r in kept if r["url"] not in existing), key=lambda r: r["url"])
    print(f"  {len(kept) - len(fresh)} already labeled or batched, dropped")

    print("\n5. Write")
    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in fresh:
            f.write(json.dumps({"url": r["url"], "hint": "jev_openweb",
                                "batch": BATCH_ID}) + "\n")
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        for r in fresh:
            f.write(json.dumps({"url": r["url"], "p_legal": r["p_legal"]}) + "\n")
    print(f"  {len(fresh)} URLs to label : {OUTPUT_FILE}")
    print(f"  scores held separately: {SCORES_FILE}")

    print("\n6. Compact")
    compact(TEXT_CACHE, {r["url"] for r in kept},
            {r["url"]: r["p_legal"] for r in drawn_scored})

    print("\nsummary")
    print(f"  drawn {len(rows):,}  with text {with_text:,}  scored {len(drawn_scored):,}"
          f"  kept @{CUT} {len(kept)}  to label {len(fresh)}")
    print(f"  Jev: {len(tokens):,} requests, {total_tokens:,} input tokens, "
          f"~${total_tokens / 1e6 * PRICE_PER_M:.2f}")
    n = len(fresh)
    print(f"  if all {n} come back legal, precision >= {wilson_lower(n, n):.3f} (95% lower bound);"
          f" one non-legal -> {wilson_lower(max(0, n - 1), n):.3f}")


if __name__ == "__main__":
    main()
"""
Fetches N_DRAW pages of the open crawl, asks Jev whether each one is a legal
document, and writes the ones it says yes to into a file to hand label.

The draw is uniform, so the pages are whatever the crawl actually holds
rather than anywhere law was expected to be. Each page is fetched and
extracted the way the training text was, then scored on its own. Anything at
or above CUT goes to the labeling file; everything else keeps its score on
disk and is not looked at again.

Hand labeling that kept file is what produces a number. Precision is the
share of kept pages that really are legal, so the kept set is the entire
denominator and nothing outside it has to be read. A hundred thousand pages
leaves about a hundred and sixty to label. The false positive rate cannot be
had this cheaply, since its denominator is every non-legal page in the draw,
so this answers precision and leaves that alone.

Fetching and scoring run at the same time. Scoring is much the faster of the
two, so this is not about finishing sooner: it means kept pages appear
minutes into a run, and a run killed partway through has scored everything it
fetched.

The labeling file is only ever added to. A URL already in it keeps its row
and whatever label was written there, so rerunning this never costs labeling
work. Scores go to a separate file, as in every other batch here, because the
labeling agent agreeing with Jev is the thing being tested.

No flags. The constants below are the experiment, and every stage resumes.

Reads:
  data/candidates/raw_pool.jsonl

Writes:
  data/labels/jev_openweb_scores.jsonl           every page scored
  data/candidates/jev_openweb_kept_full.jsonl    the labeling file
  data/candidates/jev_openweb_kept_scores.jsonl  score per kept URL

Label the kept file with prompts/legal_url_labeling_task.md, then read the
result with eval_is_legal.py.

Requires JEV_API_KEY and the AWS keys in .env.
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
import fetch_candidate_urls as pool_puller  # noqa: E402
import label_is_legal as jev  # noqa: E402

N_DRAW = 100_000
CUT    = 0.90
SEED   = 42   # same as fetch_flagged_urls, so a draw from the same pool is the same draw

# One doc per request. The first pass packed ~40 docs per request and Jev
# did not isolate docs[i]: an office-chair listing packed next to a Vermont
# statute came back 0.84, and statutes in packed requests scored 0.83 where
# the same kind of page alone scores 0.97. Five of 175 requests held 60% of
# the yes calls. That pass is kept as jev_openweb_scores.packed.jsonl.
STATE_TOKENS = 0

# Jev's cap is 1,200 requests/min. Single-doc requests make the cap the
# ceiling rather than fetch, so run enough workers to sit against it; the
# SDK backs off on 429 and label_is_legal's retry budget covers the burst.
JEV_WORKERS = 20

# How long the packer lets a part-filled request sit before sending it
# anyway. Fetch workers land pages at a steady clip, so this only fires at
# the tail of the run or if the crawl is throttling.
IDLE_FLUSH_S = 5.0

POOL_FILE     = "data/candidates/raw_pool.jsonl"
TEXT_CACHE    = "data/candidates/_pool_text_cache.jsonl"
POINTER_CACHE = "data/candidates/_pool_pointers_cache.jsonl"
JEV_SCORES    = "data/labels/jev_openweb_scores.jsonl"
JEV_SKIPPED   = "data/labels/jev_openweb_scores.skipped.jsonl"
KEPT_FULL     = "data/candidates/jev_openweb_kept_full.jsonl"
SCORES_FILE   = "data/candidates/jev_openweb_kept_scores.jsonl"

# What an unlabeled row in KEPT_FULL holds. Anything else in a row's `label`
# is a hand label and is never touched again.
UNLABELED     = "your_label"
PRICE_PER_M   = 0.042   # USD per million input tokens, jev-latest

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
        packer = jev.Packer(STATE_TOKENS)
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

    def refetch(row):
        """A page another run's compaction stripped (fetch_flagged_urls
        leaves "not flagged, text dropped") has to be fetched again to be
        scored here. One this run compacted is already in the scores file
        and stays skipped."""
        if row.get("text") or row["url"] in already_scored:
            return False
        return needs_retry(row) or "text dropped" in (row.get("skip_reason") or "")

    todo = [(r, pointers.get(r["url"])) for r in rows
            if r["url"] not in have or refetch(have[r["url"]])]
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


def merge_kept_full(path, kept):
    """Stage 5. Fold this run's kept URLs into the hand-labeling file without
    disturbing what is already there.

    The file is the labeling worksheet and the precision denominator at once,
    so it is additive and never authoritative about anything but its own
    labels: rows keep their position and their `label` exactly as found, a URL
    that is new gets appended with the UNLABELED placeholder, and a URL that
    was kept by an earlier run but not by this one stays put. That last case
    is deliberate. Raising CUT or redrawing must not silently delete labels
    someone spent an afternoon producing; pairing the file against
    SCORES_FILE is what selects a subset for a given threshold.

    Returns (added, kept_labels).
    """
    rows, seen = [], set()
    if os.path.exists(path):
        for r in load_jsonl(path):
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            rows.append(r)
    labeled = sum(1 for r in rows if r.get("label") not in (None, UNLABELED))

    added = 0
    for r in kept:
        if r["url"] not in seen:
            seen.add(r["url"])
            rows.append({"url": r["url"], "label": UNLABELED})
            added += 1

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return added, labeled


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
    """Rewrite the cache line by line, dropping page text for every page not
    in `keep_text`. The url, skip reason and Jev score stay regardless,
    which is all a rerun needs to know the page is done. Later lines win, as
    when the cache is read, so the rewrite also removes retried duplicates.

    `keep_text` is every scored URL, not just the ones that cleared CUT: the
    whole uniform draw's text is worth keeping regardless of what Jev says
    about it, since it is a second, independent labeled sample the sklearn
    classifier could be retrained on if Jev does not pan out."""
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
        labeler = jev.Labeler(client, JEV_SCORES, JEV_SKIPPED, workers=JEV_WORKERS)
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
    # Every kept page goes out, overlap included. Precision is legal-kept
    # over kept, so a denominator missing the pages other samplers already
    # found is a denominator for a different question.
    kept.sort(key=lambda r: r["url"])

    print("\n5. Write")
    added, labeled = merge_kept_full(KEPT_FULL, kept)
    os.makedirs(os.path.dirname(SCORES_FILE) or ".", exist_ok=True)
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({"url": r["url"], "p_legal": r["p_legal"],
                                "jev_version": r.get("jev_version")}) + "\n")
    print(f"  {KEPT_FULL}: {added} new rows, {labeled} hand labels preserved")
    print(f"  scores held separately: {SCORES_FILE}")

    print("\n6. Compact")
    compact(TEXT_CACHE, {r["url"] for r in drawn_scored},
            {r["url"]: r["p_legal"] for r in drawn_scored})

    print("\nsummary")
    print(f"  drawn {len(rows):,}  with text {with_text:,}  scored {len(drawn_scored):,}"
          f"  kept @{CUT} {len(kept)}")
    print(f"  Jev: {len(tokens):,} requests, {total_tokens:,} input tokens, "
          f"~${total_tokens / 1e6 * PRICE_PER_M:.2f}")
    n = len(kept)
    print(f"  if all {n} come back legal, precision >= {wilson_lower(n, n):.3f} (95% lower bound);"
          f" one non-legal -> {wilson_lower(max(0, n - 1), n):.3f}")


if __name__ == "__main__":
    main()
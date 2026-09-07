"""
Builds a labeling batch out of the pages the classifier calls legal on real
crawl data.

Every batch of labels the model trains on was drawn from somewhere law was
expected to be: URL patterns that look legal, the court and legislature
domains CourtListener names, the bill sections Open States names. So its
negatives are near misses from legal publishers, which is a much harder and
much narrower set of pages than the crawl it runs on.

This draws a plain random sample of the crawl instead, scores every page in
it, and keeps the ones the model flags. Those pages are where the model is
wrong in the way that matters, because a page it flags is a page that would
end up in the corpus.

Labeling the flagged pages answers two questions with one pass:

  1. How often is the model right when it says legal, on real crawl data.
     Every flagged page in the sample is kept, so the labels are a fair
     sample of what the model selects, and the fraction labeled legal is the
     precision the corpus will actually have.
  2. What is it wrong about. Each flagged page that comes back non_legal is a
     training negative drawn from the exact distribution the model fails on,
     which negatives sourced from legal publishers cannot supply.

How it works:

  1. Draw. Sort the pool, shuffle it with a fixed seed, take the first --n.
     The seed makes the draw repeatable, and taking a prefix means a larger
     --n is a superset of a smaller one, so raising it reuses everything
     already fetched instead of starting over.

  2. Locate and fetch. Ask the crawl index where each page lives, then pull
     and extract it, using fetch_warc_text.py's own functions so the text is
     prepared exactly the way the training text was. Anything different is a
     different input and the scores would not mean the same thing. Results
     are cached, so a rerun costs nothing for pages already held.

  3. Score and keep. Run the saved model over the text and keep every page at
     or above --min-score. Pages already labeled, or already sent out in an
     earlier batch, are dropped so nothing is labeled twice.

Two files come out, and the split is deliberate:

  data/candidates/flagged_sample_batch.jsonl   the URLs, with no score
  data/candidates/flagged_sample_scores.jsonl  the scores, keyed by URL

The labeling agent reads the first and never sees the second. A visible score
would tell it what the model already thinks, and agreement with the model is
the one thing this batch exists to test.
"""
import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corpus"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classifier"))
from fetch_warc_text import (N_WORKERS, build_row, load_jsonl,  # noqa: E402
                             needs_retry, resolve_pointers)
from features import MODES, legal_probs, load_bundle  # noqa: E402

POOL_FILE = "data/candidates/raw_pool.jsonl"
TEXT_CACHE = "data/candidates/_pool_text_cache.jsonl"
POINTER_CACHE = "data/candidates/_pool_pointers_cache.jsonl"
OUTPUT_FILE = "data/candidates/flagged_sample_batch.jsonl"
SCORES_FILE = "data/candidates/flagged_sample_scores.jsonl"
BATCH_ID = "flagged-sample-v1"
SEED = 42

# URLs already labeled or already handed to the labeling agent, so no page is
# judged twice and no page arrives carrying an answer.
EXISTING = [
    "data/processed/labeled_urls.jsonl",
    "data/candidates/candidates.jsonl",
    "data/candidates/targeted_batch.jsonl",
    "data/candidates/host_sample_batch.jsonl",
    "data/candidates/bill_sample_batch.jsonl",
]


def draw(pool_file, n):
    """The first n of a seeded shuffle of the pool.

    Sorting before shuffling matters: a set has no defined order, so without
    the sort the same seed would draw a different sample on a different run.
    """
    urls = set()
    with open(pool_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                urls.add(json.loads(line)["url"])
    ordered = sorted(urls)
    random.Random(SEED).shuffle(ordered)
    print(f"pool: {len(ordered)} unique URLs, drawing {n}")
    return [{"url": u} for u in ordered[:n]]


def load_existing_urls(paths):
    urls = set()
    for path in paths:
        if os.path.exists(path):
            urls.update(r["url"] for r in load_jsonl(path))
    return urls


def fetch_text(rows, pointers, cache_path):
    """Fill the text cache for any drawn URL it does not already hold.

    Each page is appended to the cache the moment it lands, rather than the
    whole set being written at the end. Extraction runs in C libraries that
    can take the interpreter down on a malformed page, and a crash at page
    26000 would otherwise discard every page before it. Appending means a
    crashed run keeps everything it already paid for and the next run
    resumes.

    The cache is therefore append-only and can hold a URL more than once, so
    reading it keeps the last entry for each. It gets rewritten in one piece
    at the end of a clean run, which compacts those repeats away.
    """
    have = {}
    if os.path.exists(cache_path):
        # later lines win, so a retry supersedes the failure it replaces
        for r in load_jsonl(cache_path):
            have[r["url"]] = r
        print(f"  cached: {len(have)} pages")

    todo = [(r, pointers.get(r["url"])) for r in rows
            if r["url"] not in have or needs_retry(have[r["url"]])]
    print(f"  {len(todo)} to fetch")
    if todo:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        lock = threading.Lock()
        done = 0
        with open(cache_path, "a", encoding="utf-8") as cache:

            def guarded(item):
                """Claim a URL before touching it.

                A crash inside the extractor cannot be caught, so a page that
                causes one would be first in line on the next run and crash
                that run too, and no run would ever get past it. Writing the
                claim first means a page that never came back is on disk as
                failed, and the next run skips it instead of dying on it.
                """
                with lock:
                    cache.write(json.dumps(
                        {"url": item[0]["url"], "text": None,
                         "skip_reason": "no result, extraction did not return"}) + "\n")
                    cache.flush()
                return build_row(item)

            with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                for row in pool.map(guarded, todo):
                    with lock:
                        have[row["url"]] = row
                        cache.write(json.dumps(row, ensure_ascii=False) + "\n")
                        done += 1
                        if done % 500 == 0:
                            cache.flush()
                            print(f"  {done}/{len(todo)}")
        with open(cache_path, "w", encoding="utf-8") as f:
            for url in sorted(have):
                f.write(json.dumps(have[url], ensure_ascii=False) + "\n")
        print(f"  compacted {cache_path} ({len(have)} pages)")
    return have


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--n", type=int, default=32000,
                    help="pages to draw from the pool. At a flag rate near "
                         "0.6 percent, 32000 yields roughly 200 flagged "
                         "pages, which is enough to measure precision to "
                         "about plus or minus 6 points.")
    ap.add_argument("--model", default="models/text_clf.joblib")
    ap.add_argument("--min-score", type=float, default=0.5,
                    help="keep pages scoring at or above this. Set below the "
                         "operating threshold on purpose, so the labels also "
                         "cover the band just under it and the threshold can "
                         "be chosen from the results rather than fixed first.")
    ap.add_argument("--output", default=OUTPUT_FILE)
    ap.add_argument("--scores", default=SCORES_FILE)
    args = ap.parse_args()

    rows = draw(POOL_FILE, args.n)

    print("\n1. WARC pointers")
    pointers = resolve_pointers(rows, POINTER_CACHE)

    print("\n2. Page text")
    have = fetch_text(rows, pointers, TEXT_CACHE)

    print("\n3. Scoring")
    bundle = load_bundle(args.model)
    field = MODES[bundle["mode"]][2]
    drawn = [have[r["url"]] for r in rows if r["url"] in have]
    scored = [r for r in drawn if r.get(field)]
    print(f"  {len(scored)} of {len(rows)} drawn pages have text")

    probs = legal_probs(bundle, [r[field] for r in scored])
    flagged = [(r, float(p)) for r, p in zip(scored, probs) if p >= args.min_score]
    rate = len(flagged) / len(scored)
    print(f"  {len(flagged)} at or above {args.min_score} "
          f"({rate:.4%} of pages with text)")

    existing = load_existing_urls(EXISTING)
    fresh = [(r, p) for r, p in flagged if r["url"] not in existing]
    print(f"  {len(flagged) - len(fresh)} already labeled or batched, dropped")

    fresh.sort(key=lambda rp: rp[0]["url"])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r, _ in fresh:
            f.write(json.dumps({"url": r["url"],
                                "hint": "model_flagged",
                                "batch": BATCH_ID}) + "\n")
    with open(args.scores, "w", encoding="utf-8") as f:
        for r, p in fresh:
            f.write(json.dumps({"url": r["url"],
                                "prob_legal": round(p, 6)}) + "\n")

    print(f"\n{len(fresh)} URLs written: {args.output}")
    print(f"scores held separately : {args.scores}")
    print(f"\n{'band':>12}  pages")
    bands = [(0.5, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]
    for lo, hi in bands:
        n = sum(1 for _, p in fresh if lo <= p < hi)
        print(f"{lo:.2f} to {hi:.2f}  {n:>5}")


if __name__ == "__main__":
    main()

"""
Labels the pages source_pages.py kept as legal or general, using the frozen
classifier cascade. Writes only the labels; the text stays in pages/.

Three workers, joined by queues held in memory:

  tfidf   a pool of processes (--screen-workers), each taking one finished
          pages/ file at a time and scoring its pages with the TF-IDF model
          (models/text_clf.joblib). Below 0.30: general, straight to the
          writer. At or above: to the Laya queue
  laya    one thread (one GPU): scores queued pages in batches with Laya
          (models/laya-legal, 1024 tokens); legal at p >= 0.85, general below;
          results go to the writer
  writer  one thread, the only one that writes: appends one line per page to
          labels/<file>.jsonl, the pages/ file the page came from

So TF-IDF never waits on Laya; only pages that pass it do. Questions and page
format come from classifier/largeModel/score_laya.py, so a page gets the same
score here as in the classifier's evaluations.

A pages/ file is picked up once source_pages.py has finished it (its stats file
exists). Pages that already have a label line are skipped, so a rerun resumes;
pages still in a queue when the run stops are labeled again next time. With
--watch, the script keeps checking for new files while source_pages.py runs,
and stops once source_pages.py has finished and every page is labeled.

Runs in the laya environment (torch + laya, plus scikit-learn 1.9.0, joblib and
tldextract for TF-IDF), not the project venv.

Usage:
  C:\\projects\\laya-env\\Scripts\\python.exe src/corpus/sourcing/label_pages.py --out data/source [--watch]

Outputs (--out):
  labels/<file>.jsonl   one line per page: id, tfidf (TF-IDF score), p_legal
                        (Laya's score; null if TF-IDF screened it out) and
                        bucket ("legal" | "general")

Next:
  python src/corpus/dedup/find_pairs.py --pages data/source/pages --labels data/source/labels --out data/dedup
"""
import argparse
import glob
import gzip
import json
import os
import queue
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("../../classifier/largeModel", "../../classifier/model"):
    sys.path.insert(0, os.path.join(HERE, sub))
from score_laya import QUESTIONS, TFIDF_MODEL, state_for  # noqa: E402

SCREEN  = 0.30                  # frozen TF-IDF cut
KEEP    = 0.85                  # frozen Laya cut
MODEL   = "models/laya-legal"
MAX_LEN = 1024
POLL    = 10                    # seconds between checks for new files and progress lines
DONE    = None                  # queue item telling a thread to finish


# rows of a .jsonl or .jsonl.gz file
def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue        # a line cut off by a crash; that page is labeled again


# pages/ files source_pages.py has finished: name -> how many pages it kept
def finished_files(out, known):
    for p in glob.glob(f"{out}/stats/*.json"):
        name = os.path.basename(p).removesuffix(".json")
        if name not in known:
            with open(p, encoding="utf-8") as f:
                known[name] = json.load(f).get("kept", 0)
    return known


# ids already in a file's labels, read once at startup
def labeled_ids(out, name):
    path = f"{out}/labels/{name}.jsonl"
    return {r["id"] for r in read_jsonl(path)} if os.path.exists(path) else set()


# the TF-IDF model, loaded once per worker process
_tfidf = {}


def tfidf_init():
    from features import legal_probs, load_bundle
    _tfidf.update(bundle=load_bundle(TFIDF_MODEL, quiet=True), probs=legal_probs)


# score one pages/ file's unlabeled pages: (file, general [(id, tfidf)], for Laya [(id, tfidf, url, text)])
def tfidf_file(out, name, done):
    rows = [r for r in read_jsonl(f"{out}/pages/{name}.jsonl.gz") if r["id"] not in done]
    scores = _tfidf["probs"](_tfidf["bundle"], [r["text"] for r in rows]) if rows else []
    general, to_laya = [], []
    for r, p in zip(rows, scores):
        p = round(float(p), 4)
        if p >= SCREEN:
            to_laya.append((r["id"], p, r["url"], r["text"]))
        else:
            general.append((r["id"], p))
    return name, general, to_laya


# laya thread: score queued pages in GPU batches, send labels to the writer
def laya_worker(agent, batch_size, laya_q, write_q, counts):
    stop = False
    while not stop:
        batch, item = [], laya_q.get()           # wait for work, then take what's queued
        while item is not DONE:
            batch.append(item)
            if len(batch) >= batch_size * 8:
                break
            try:
                item = laya_q.get_nowait()
            except queue.Empty:
                break
        stop = item is DONE
        if not batch:
            continue
        states = [state_for(url, text) for _, _, _, url, text in batch]
        results = agent.predict_batch(states, QUESTIONS, batch_size=batch_size,
                                      max_len=MAX_LEN, sort_by_length=len(states) > batch_size)
        for (name, pid, tfidf, _, _), res in zip(batch, results):
            p = round(res["answers"]["is_legal"]["noul"], 4)
            write_q.put((name, [(pid, tfidf, p, "legal" if p >= KEEP else "general")]))
        counts["laya done"] += len(batch)
    write_q.put(DONE)


# writer thread, the only one that writes: append each label line to its file
def write_worker(out, write_q, counts):
    while (item := write_q.get()) is not DONE:
        name, labels = item
        with open(f"{out}/labels/{name}.jsonl", "a", encoding="utf-8") as f:
            for pid, tfidf, p, bucket in labels:
                f.write(json.dumps({"id": pid, "tfidf": tfidf, "p_legal": p, "bucket": bucket}) + "\n")
        counts["written"] += len(labels)
        counts["legal"] += sum(1 for *_, b in labels if b == "legal")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--out", default="data/source", help="source_pages.py's --out folder")
    ap.add_argument("--screen-workers", type=int, default=4, help="TF-IDF processes")
    ap.add_argument("--batch-size", type=int, default=8, help="pages per GPU pass; lower if it runs out of memory")
    ap.add_argument("--watch", action="store_true", help="keep labeling new files until source_pages.py is done")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/labels", exist_ok=True)

    import laya                                       # heavy; loaded once
    agent = laya.load(os.path.normpath(MODEL))
    print(f"Laya {MODEL} on {agent.device}, max_len {MAX_LEN}, batch {args.batch_size}")

    counts, laya_q, write_q = Counter(), queue.Queue(), queue.Queue()
    threads = [threading.Thread(target=laya_worker, args=(agent, args.batch_size, laya_q, write_q, counts)),
               threading.Thread(target=write_worker, args=(args.out, write_q, counts))]
    for t in threads:
        t.start()

    files, sent, running = {}, set(), []
    with ProcessPoolExecutor(args.screen_workers, initializer=tfidf_init) as pool:
        while True:
            # hand each finished pages/ file to TF-IDF once per run, minus pages already labeled
            for name, kept in sorted(finished_files(args.out, files).items()):
                if name not in sent:
                    sent.add(name)
                    done = labeled_ids(args.out, name)
                    if len(done) < kept:
                        running.append(pool.submit(tfidf_file, args.out, name, done))
            # route finished TF-IDF results: general to the writer, the rest to Laya
            for job in [j for j in running if j.done()]:
                running.remove(job)
                name, general, to_laya = job.result()
                if general:
                    write_q.put((name, [(pid, p, None, "general") for pid, p in general]))
                for pid, p, url, text in to_laya:
                    laya_q.put((name, pid, p, url, text))
                counts["screened"] += len(general) + len(to_laya)
                counts["to laya"] += len(to_laya)
            print(f"  screened {counts['screened']:,} | to Laya {counts['to laya']:,} "
                  f"(waiting {laya_q.qsize():,}) | written {counts['written']:,}, {counts['legal']:,} legal")

            # done when no TF-IDF work is left and (with --watch) sourcing has finished too
            source_done = not args.watch or os.path.exists(f"{args.out}/source.done")
            if not running and source_done and set(finished_files(args.out, files)) <= sent:
                break
            time.sleep(POLL)

    laya_q.put(DONE)                                  # Laya drains its queue, then stops the writer
    for t in threads:
        t.join()
    n, legal = counts["written"], counts["legal"]
    print(f"\nthis run: {n:,} pages labeled, {counts['to laya']:,} sent to Laya, {legal:,} legal"
          + (f" ({legal / n:.2%})" if n else ""))


if __name__ == "__main__":
    main()
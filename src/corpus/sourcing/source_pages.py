"""
Pulls pages from Common Crawl for the full run: picks random WARC files from
every crawl segment, and turns each file into the pages that pass the quality
filters, as text.

Picking files. CC-MAIN-2026-12 has 100 segments of 1,000 WARC files each. Each
segment's files are shuffled in a fixed order (same seed every run) and the
first --per-segment are taken, so every segment gives the same number of files,
and a later run with a larger --per-segment only adds files.

Whole files are read, not the first pages of each: records inside a WARC file
are in alphabetical URL order, so the front of a file is not a random sample.

One worker per file streams it from the Common Crawl S3 bucket and, for each
page:

  1. scope     HTTP status 200, Common Crawl's detected type HTML, and English
               the only language Common Crawl's detector found (a page with no
               language found, nearly always one with almost no text, is out).
               Pages outside it are counted, not logged
  2. url       the URL filter (filters/url.py), before extraction, so a dropped
               page is never extracted
  3. extract   trafilatura, with fetch_warc_text.py's pinned settings, so text
               matches what the classifier was trained on
  4. filters   Gopher, then C4 (filters/pipeline.py)

Needs AWS credentials in .env (the bucket rejects unsigned requests; nothing is
billed). A file whose outputs exist is skipped, so a rerun resumes.

Usage:
  python src/corpus/sourcing/source_pages.py --per-segment 1 --out data/source

Outputs (--out), one of each per WARC file:
  pages/<file>.jsonl.gz     kept pages: id (dedup/minhash.page_id), url,
                            text, warc_filename, warc_record_offset,
                            warc_record_length
  dropped/<file>.jsonl.gz   dropped pages: the same, without text, plus reason
                            (e.g. "url: blocked domain", "gopher: word count",
                            "extract: empty", or "error: <type>" for a page
                            that raised an unexpected error)
  stats/<file>.json         counts per stage and seconds taken; written last,
                            so it marks the file done
  warc.paths.gz             the snapshot's list of WARC files
  source.done               written when the run ends, so label_pages.py
                            --watch knows to stop

Next (in the laya environment):
  C:\\projects\\laya-env\\Scripts\\python.exe src/corpus/sourcing/label_pages.py --out data/source
"""
import argparse
import gzip
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import timedelta
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager

import trafilatura
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from warcio.archiveiterator import ArchiveIterator

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("../../common", "../filters", "../dedup"):
    sys.path.insert(0, os.path.join(HERE, sub))
from fetch_warc_text import _s3, CC_BUCKET, EXTRACT_OPTS, MAX_TEXT_CHARS, SNAPSHOT  # noqa: E402
from minhash import page_id  # noqa: E402
from pipeline import page_reject  # noqa: E402
from url import url_reject  # noqa: E402

SEED      = 42
HTML      = ("text/html", "application/xhtml+xml")   # detected types that count as HTML
RETRIES   = 3                                         # whole-file attempts before giving up
PATHS_KEY = f"crawl-data/{SNAPSHOT}/warc.paths.gz"
REPORT    = 0.5                                       # seconds between a worker's progress updates
RESUMES   = 5                                         # reconnects per read before the file is retried whole


# one WARC file's download, counting bytes read; if the connection drops, it
# reconnects and continues from the byte where it stopped, so nothing is re-read
class Download:
    def __init__(self, key):
        self.key, self.n, self.resumes = key, 0, 0
        obj = _s3.get_object(Bucket=CC_BUCKET, Key=key)
        self.size, self.body = obj["ContentLength"], obj["Body"]

    def read(self, size=-1):
        for attempt in range(RESUMES):
            try:
                data = self.body.read(size)
                self.n += len(data)
                return data
            except Exception:
                if attempt == RESUMES - 1:
                    raise
                time.sleep(2 ** attempt + random.random())
                self.body = _s3.get_object(Bucket=CC_BUCKET, Key=self.key,
                                           Range=f"bytes={self.n}-")["Body"]
                self.resumes += 1


# the snapshot's WARC file list, downloaded once
def warc_paths(out):
    path = f"{out}/warc.paths.gz"
    if not os.path.exists(path):
        _s3.download_file(CC_BUCKET, PATHS_KEY, path)
    with gzip.open(path, "rt") as f:
        return [line.strip() for line in f if line.strip()]


# the first n files of each segment's fixed shuffle
def pick_files(paths, n):
    by_segment = defaultdict(list)
    for p in paths:
        by_segment[p.split("/")[3]].append(p)       # crawl-data/<snapshot>/segments/<segment>/warc/...
    picked = []
    for seg in sorted(by_segment):
        files = sorted(by_segment[seg])
        random.Random(f"{SEED}-{seg}").shuffle(files)
        picked += files[:n]
    return picked


# a WARC file's name without folders or .warc.gz
def stem(key):
    return os.path.basename(key).removesuffix(".warc.gz")


# English is the only language CLD2 detected, as in the index's content_languages = 'eng'
def english_only(metadata):
    for line in metadata.decode("utf-8", "replace").splitlines():
        if line.startswith("languages-cld2:"):
            # no "languages" list when CLD2 found too little text to judge
            langs = json.loads(line.split(":", 1)[1]).get("languages", [])
            return [l.get("code-iso-639-3") for l in langs] == ["eng"]
    return False


# extract and filter one in-scope page: (row to keep, None) or (None, reason)
def judge(url, html):
    if reason := url_reject(url):
        return None, f"url: {reason}"
    try:
        text = trafilatura.extract(html, **EXTRACT_OPTS)
    except Exception as exc:
        return None, f"extract: {type(exc).__name__}"
    if not text or not text.strip():
        return None, "extract: empty"
    text = text[:MAX_TEXT_CHARS]
    if reason := page_reject(url, text):
        return None, reason
    return text, None


# one pass over one WARC file, writing to the open pages/dropped files and
# reporting (bytes read, file size, pages kept) to progress[name]; returns counts
def read_file(key, pages, dropped, name, progress):
    counts = Counter()
    pending = {}            # response record ID -> page waiting for its metadata record
    stream, last = Download(key), 0.0
    it = ArchiveIterator(stream)
    for rec in it:
        if time.monotonic() - last >= REPORT:
            progress[name] = (stream.n, stream.size, counts["kept"])
            last = time.monotonic()
        h = rec.rec_headers
        if rec.rec_type == "response":
            counts["responses"] += 1
            in_scope = (rec.http_headers is not None
                        and rec.http_headers.get_statuscode() == "200"
                        and h.get_header("WARC-Identified-Payload-Type") in HTML)
            # read the page before asking for its offset: asking first leaves the page empty (warcio)
            html = rec.content_stream().read() if in_scope else None
            it.read_to_end(rec)
            if in_scope:
                pending[h.get_header("WARC-Record-ID")] = (
                    h.get_header("WARC-Target-URI"), html, it.get_record_offset(), it.get_record_length())
            else:
                counts["out of scope"] += 1
        elif rec.rec_type == "metadata":
            page = pending.pop(h.get_header("WARC-Concurrent-To"), None)
            if page is None:
                continue
            url, html, offset, length = page
            metadata = rec.content_stream().read()      # outside the try: a download error retries the file
            row = {"id": page_id(url), "url": url, "warc_filename": key,
                   "warc_record_offset": offset, "warc_record_length": length}
            try:
                if not english_only(metadata):
                    counts["out of scope"] += 1
                    continue
                text, reason = judge(url, html)
            except Exception as exc:                    # one odd page is dropped, not the whole file
                text, reason = None, f"error: {type(exc).__name__}"
            if reason:
                counts[f"dropped: {reason}"] += 1
                dropped.write(json.dumps({**row, "reason": reason}, ensure_ascii=False) + "\n")
            else:
                counts["kept"] += 1
                pages.write(json.dumps({**row, "text": text}, ensure_ascii=False) + "\n")
    counts["out of scope"] += len(pending)      # responses that never got a metadata record
    counts["reconnects"] = stream.resumes
    counts["bytes"] = stream.size
    return counts


# one WARC file, start to finish; retried whole if the download breaks
def process_file(key, out, progress):
    name = stem(key)
    done = f"{out}/stats/{name}.json"
    if os.path.exists(done):
        return name, None
    paths = {d: f"{out}/{d}/{name}.jsonl.gz" for d in ("pages", "dropped")}
    tmp = {d: p + ".tmp" for d, p in paths.items()}
    start = time.time()
    try:
        for attempt in range(RETRIES):
            try:
                with gzip.open(tmp["pages"], "wt", encoding="utf-8") as pages, \
                     gzip.open(tmp["dropped"], "wt", encoding="utf-8") as dropped:
                    counts = read_file(key, pages, dropped, name, progress)
                break
            except Exception as exc:
                if attempt == RETRIES - 1:
                    return name, f"failed: {type(exc).__name__}: {exc}"
                time.sleep(5 * 2 ** attempt + random.random())
    finally:
        progress.pop(name, None)                 # its bar disappears
    for d in paths:
        os.replace(tmp[d], paths[d])
    stats = {"warc": key, "seconds": round(time.time() - start, 1), **counts}
    with open(done + ".tmp", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    os.replace(done + ".tmp", done)
    return name, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--per-segment", type=int, default=1, help="WARC files per crawl segment")
    ap.add_argument("--out", default="data/source")
    ap.add_argument("--workers", type=int, default=2 * os.cpu_count(),
                    help="files processed at once; above the core count since downloads wait on the network")
    args = ap.parse_args()
    start = time.monotonic()                     # wall clock for the run, printed at the end

    for d in ("pages", "dropped", "stats"):
        os.makedirs(f"{args.out}/{d}", exist_ok=True)
    marker = f"{args.out}/source.done"           # tells label_pages.py --watch this run has finished
    if os.path.exists(marker):
        os.remove(marker)
    files = pick_files(warc_paths(args.out), args.per_segment)
    todo = [k for k in files if not os.path.exists(f"{args.out}/stats/{stem(k)}.json")]
    print(f"{len(files):,} WARC files ({args.per_segment} per segment), "
          f"{len(files) - len(todo):,} already done, {len(todo):,} to go")

    url_reject("http://example.com/")    # fetch the URL blocklists once, before workers race for them

    totals, failed, finished = Counter(), [], 0
    sizes, done_bytes = {}, 0                    # file name -> size, for files seen so far; bytes of finished files
    columns = (TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
               TextColumn("{task.fields[info]}"), TimeRemainingColumn())
    # speed averaged over 10 minutes, so files finishing in bursts don't swing the estimate
    with Manager() as manager, ProcessPoolExecutor(args.workers) as ex, \
         Progress(*columns, speed_estimate_period=600) as bars:
        progress = manager.dict()                # file name -> (bytes read, file size, pages kept)
        jobs = {ex.submit(process_file, k, args.out, progress) for k in todo}
        overall = bars.add_task(f"files 0/{len(todo)}", total=None, info="")
        rows = {}                                # file name -> its bar
        while jobs:
            done, jobs = wait(jobs, timeout=REPORT, return_when=FIRST_COMPLETED)
            for job in done:
                finished += 1
                name, stats = job.result()
                if isinstance(stats, str):
                    failed.append(name)
                    done_bytes += sum(sizes.values()) / len(sizes) if sizes else 0   # count it as one average file
                    bars.console.print(escape(f"  [{finished}/{len(todo)}] {name}: {stats}"))
                    continue
                totals.update({k: v for k, v in stats.items() if k != "warc"})
                sizes[name] = stats["bytes"]
                done_bytes += stats["bytes"]
                bars.console.print(escape(f"  [{finished}/{len(todo)}] {name}: {stats.get('kept', 0):,} kept of "
                                          f"{stats.get('responses', 0):,} pages in {stats['seconds']:.0f}s"))
            # one bar per file being read, showing how far through its download it is
            now = dict(progress)
            for name, (read, size, kept) in now.items():
                if name not in rows:
                    # start time + file number (e.g. 193222-00141); the number alone repeats across segments
                    rows[name] = bars.add_task(f"{name[16:22]}-{name[-5:]}", total=size, info="")
                    sizes[name] = size
                bars.update(rows[name], completed=read, total=size,
                            info=f"{read / 1e6:,.0f}/{size / 1e6:,.0f} MB  {kept:,} kept")
            for name in [n for n in rows if n not in now]:
                bars.remove_task(rows.pop(name))
            # overall, by bytes: finished files plus what running files have read, out of
            # (files to do x average file size seen so far)
            if sizes:
                avg = sum(sizes.values()) / len(sizes)
                bars.update(overall, description=f"files {finished}/{len(todo)}",
                            completed=done_bytes + sum(r for r, _, _ in now.values()),
                            total=len(todo) * avg,
                            info=f"{totals['kept']:,} kept")

    if totals:
        print(f"\nthis run: {totals['responses']:,} pages, {totals['out of scope']:,} out of scope, "
              f"{totals['kept']:,} kept")
        for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
            if k.startswith("dropped"):
                print(f"  {v:>10,}  {k}")
    if failed:
        print(f"\n{len(failed)} files failed; rerun to retry them")
    print(f"took {timedelta(seconds=round(time.monotonic() - start))}")
    open(marker, "w").close()


if __name__ == "__main__":
    main()

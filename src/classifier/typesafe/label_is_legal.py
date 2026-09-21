"""
Asks Jev (TypeSafe's System One model) the one question this project cares
about, "is this page a legal document?", for every row of a text JSONL, and
writes the yes/no answer per URL.

The definition is from prompts/legal_url_labeling_task.md, lifted verbatim
so Jev is judging against exactly the same bar the hand-labeled ground truth
was produced under. If that definition changes, change it there and here
together.

Docs are packed several to a request: the definition goes into state once,
each doc goes in as `docs[i]`, and one Noul question per doc points at its
own `docs[i]`. All questions in a request run in parallel server-side. How
many docs share a request is set by a token budget on the state, not a
fixed count, so it adapts to page sizes and stays under Jev's 32k-token
state limit. The docs' own throughput pattern is one doc per request with
client-side concurrency, so packing is a cost/accuracy experiment, not the
blessed path: --state-tokens 0 gets the single-doc behaviour back.

Requests go out over a thread pool. When a packed request fails, its docs
are split in half and both halves go back on the pool rather than being
retried one at a time in order: a single bad page is isolated in log2(n)
rounds while the good pages around it get answered, and none of it waits
on the rest of the run. A request of one doc that fails is written to the
skipped file and left there. An auth or permission error stops the run
outright instead of being retried per request.

This script only batches the queries. It does not check the answers against
existing labels; that's eval_is_legal.py, so the two concerns stay apart.
The core is importable: openweb_precision.py drives the same `run()` over a
uniform crawl draw.

Usage:
    python src/classifier/typesafe/label_is_legal.py
    python src/classifier/typesafe/label_is_legal.py --input some/other.jsonl --output out.jsonl
    python src/classifier/typesafe/label_is_legal.py --state-tokens 0   # one doc per request

Input:  JSONL with at least {"url": ..., "text": ...}
        default data/processed/labeled_text.jsonl
Output: JSONL of {"url", "is_legal": "yes"|"no", "p_legal",
                  "request", "request_size", "input_tokens", "output_tokens"}
        default data/labels/jev_is_legal.jsonl
        p_legal is Jev's raw P(yes); is_legal is that cut at YES_THRESHOLD.
        Token counts are for the whole request the row rode in, not the row
        alone: to cost a run, sum tokens over distinct `request` values.
        Rows with no text or a failed request go to <output>.skipped.jsonl
        with a reason, never into the main file.

Resumable: URLs already in the output file are skipped on rerun. Every
request's answers are appended to disk as soon as they come back.

Requires JEV_API_KEY in .env.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from typesafe_sdk import (Noul, RetryPolicy, TypeSafeAuthenticationError,
                          TypeSafeClient, TypeSafeError,
                          TypeSafePermissionDeniedError)

load_dotenv()

DEFAULT_INPUT  = "data/processed/labeled_text.jsonl"
DEFAULT_OUTPUT = "data/labels/jev_is_legal.jsonl"

# Noul returns P(yes). 0.5 is the neutral cut: this script's job is just the
# label, so no reason to bias it either way. The raw probability is written
# alongside so the threshold can be revisited without rerunning.
YES_THRESHOLD = 0.5

# The API has no documented state-size limit, but the corpus has 500k-char
# outliers and the median page is ~1.3k. Cap what gets sent so one giant page
# doesn't blow the request; the first N chars are plenty to tell primary
# source text from an index/commentary page.
DEFAULT_MAX_CHARS = 20_000

# Jev allows 64k tokens per request, 32k of which is state + the longest
# question (docs.typesafe.ai/models). Docs are packed into a request until
# the estimated state size would pass this budget, which leaves headroom for
# the definition, the JSON framing, and a rough chars-to-tokens estimate.
DEFAULT_STATE_TOKEN_BUDGET = 28_000

# Fitted on the first full run (761 requests): input tokens = 0.248*chars +
# 200*docs + 904, so 4.04 chars/token, ~200 tokens of question + criteria
# per doc, ~900 of definition + framing per request.
CHARS_PER_TOKEN = 4

# Concurrent requests in flight. Same as fetch_warc_text's fetch pool; Jev's
# limit is 1,200 requests/min, so this is nowhere near it.
N_WORKERS = 6

# Same as the SDK default policy but with one more attempt, since a packed
# request that dies on a transient error takes several docs with it.
RETRY = RetryPolicy(max_retries=3)

# --- the question --------------------------------------------------------
# Verbatim from prompts/legal_url_labeling_task.md, "The definition". Only
# the URL-string framing is dropped, since Jev sees the page text directly.
# It goes into state once per request under `definition`; each per-doc
# question refers back to it rather than repeating it N times.

DEFINITION = (
    "LEGAL = the URL is from a source whose primary function is producing or "
    "publishing formal legal documents (court systems, legislative bodies, "
    "regulatory agencies, statute repositories, established legal publishers) "
    "AND the specific page's own HTML contains the actual text of a filing, "
    "statute, bill, regulation, or court opinion - not a page that merely links "
    "out to that text.\n\n"
    "Common Crawl only captures a page's own HTML, never the content behind its "
    "links. A landing/index page that links to a PDF or a separate full-text "
    "page has no legal text on the page itself, so it is NON_LEGAL even if it "
    "sits one click away from the real document and even if the linked document "
    "would itself be LEGAL.\n\n"
    "NON_LEGAL (excluded even if legal-adjacent) = legal commentary, law firm "
    "marketing pages, legal news, homepage/search/index/menu pages of legal "
    "databases, and landing/index pages that merely link to the actual "
    "document text elsewhere - even on an otherwise-qualifying domain.\n\n"
    "Example: a Justia case-listing index page is NON_LEGAL. A specific Justia "
    "opinion page is LEGAL. Same domain, different page type, different label.\n\n"
    "Example: a Justice Laws Canada PITIndex.html / index.html / "
    "FullText.html-adjacent landing page that links out to the regulation's "
    "actual text in a separate file is NON_LEGAL - only the page that contains "
    "the regulation's text itself is LEGAL.\n\n"
    "Other worked examples:\n"
    "- https://www.law.cornell.edu/uscode/text/17/107 (actual statute text) -> LEGAL\n"
    "- https://www.law.cornell.edu/ (LII homepage, navigation only) -> NON_LEGAL\n"
    "- https://www.uscourts.gov/ (bare homepage) -> NON_LEGAL\n"
    "- A law firm's blog post explaining fair use -> NON_LEGAL (commentary/marketing)\n"
    "- A news article about a court ruling -> NON_LEGAL (legal news, not the ruling itself)\n"
    "- A specific bill's full text on a legislature's site -> LEGAL\n\n"
    "If it reads like the primary source text (statute language, docket number "
    "and opinion text, bill text with section numbers), it's LEGAL. If it reads "
    "like someone describing, summarizing, or indexing that content, it's "
    "NON_LEGAL."
)

CRITERIA = {
    "true":  "LEGAL: the page's own text is the primary source - a statute, "
             "bill, regulation, filing, or court opinion - from a formal legal "
             "publisher.",
    "false": "NON_LEGAL: anything else, including commentary, news, law firm "
             "marketing, and index/landing/homepage pages on legal sites that "
             "only link to the document rather than containing it.",
}


def question_for(i):
    """One Noul aimed at docs[i]. Each doc is judged on its own; the other
    docs in the request are unrelated pages and must not influence it."""
    return Noul(
        instructions=(
            f"Apply `definition` to the page at `docs[{i}]`, judging from its "
            f"`docs[{i}].url` and `docs[{i}].text` together. `text` is the "
            f"page's own extracted content and is the only content that "
            f"counts. Is `docs[{i}]` a formal legal document as defined? "
            f"Ignore every other entry in `docs`; they are unrelated pages."
        ),
        criteria=CRITERIA,
    )


# --- io -------------------------------------------------------------------

def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def already_done(path):
    if not os.path.exists(path):
        return set()
    return {row["url"] for row in iter_jsonl(path)}


def append_all(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def est_tokens(doc):
    return (len(doc["url"]) + len(doc["text"])) // CHARS_PER_TOKEN


def pack(docs, token_budget):
    """Greedy packing in input order: a request takes docs until the next one
    would push the estimated state past the budget. A single doc over budget
    on its own still goes out alone; max_chars is what actually bounds that.
    A budget of 0 puts every doc in its own request."""
    definition_tokens = len(DEFINITION) // CHARS_PER_TOKEN
    batch, used = [], definition_tokens
    for doc in docs:
        t = est_tokens(doc)
        if batch and used + t > token_budget:
            yield batch
            batch, used = [], definition_tokens
        batch.append(doc)
        used += t
    if batch:
        yield batch


# --- querying -------------------------------------------------------------

def ask(client, docs):
    """One request for a list of {url, text} docs. Returns the parallel list
    of P(yes), plus the response's usage."""
    state = {"definition": DEFINITION, "docs": docs}
    questions = {f"is_legal_{i}": question_for(i) for i in range(len(docs))}
    resp = client.system_one(state=state, questions=questions)
    probs = [resp.nouls[f"is_legal_{i}"].noul for i in range(len(docs))]
    return probs, resp.usage


class Packer:
    """Fills requests by token budget from docs handed over one at a time.

    `add` returns a full batch when the next doc wouldn't fit, `flush`
    returns whatever is waiting. A budget of 0 makes every doc its own
    batch. This is `pack` for a stream: the labeler's caller decides when
    an idle stream is worth flushing early."""

    def __init__(self, token_budget):
        self.budget = token_budget
        self.definition_tokens = len(DEFINITION) // CHARS_PER_TOKEN
        self.batch, self.used = [], self.definition_tokens

    def add(self, doc):
        t = est_tokens(doc)
        out = None
        if self.batch and self.used + t > self.budget:
            out = self.flush()
        self.batch.append(doc)
        self.used += t
        return out

    def flush(self):
        out, self.batch, self.used = self.batch, [], self.definition_tokens
        return out or None


class Labeler:
    """Sends packed batches to Jev over a thread pool as they are submitted.

    A request that fails is split in half and both halves are resubmitted,
    so they run alongside everything else instead of one after another; a
    single doc that fails on its own is skipped. Request ids are handed out
    at submit time, so they are stable whatever order the answers come back
    in. Answer rows go to `output` and failures to `skipped` as they land.

    `close` waits for every request, including the halves spawned by
    failures, and returns the run's counts. An auth/permission error stops
    the run: nothing new is sent and `close` raises it.
    """

    def __init__(self, client, output, skipped, *, workers=N_WORKERS, log=print,
                 total=None):
        self.client, self.output, self.skipped, self.log = client, output, skipped, log
        self.total = total
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.Lock()
        self.idle = threading.Condition(self.lock)
        self.inflight = 0
        self.next_id = 0
        self.fatal = None
        self.stats = {"yes": 0, "no": 0, "skipped": 0, "done": 0, "requests": 0,
                      "bisects": 0, "input_tokens": 0, "output_tokens": 0}
        self.t0 = time.time()
        self._last_report = 0

    def submit(self, docs):
        with self.lock:
            if self.fatal:
                return
            req_id = self.next_id
            self.next_id += 1
            self.inflight += 1
        self.pool.submit(self._work, docs, req_id)

    def _work(self, docs, req_id):
        try:
            self._query(docs, req_id)
        except BaseException as e:   # a bug, not an API error: surface it, don't hang close()
            with self.lock:
                self.fatal = self.fatal or e
        finally:
            with self.lock:
                self.inflight -= 1
                if self.inflight == 0:
                    self.idle.notify_all()

    def _query(self, docs, req_id):
        if self.fatal:
            return
        try:
            probs, usage = ask(self.client, docs)
        except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError) as e:
            with self.lock:
                self.fatal = e
            return
        except TypeSafeError as e:
            if len(docs) == 1:
                with self.lock:
                    append_all(self.skipped, [{"url": docs[0]["url"],
                                               "reason": f"{type(e).__name__}: {e}"}])
                    self.stats["skipped"] += 1
                    self.stats["done"] += 1
                return
            mid = len(docs) // 2
            with self.lock:
                self.stats["bisects"] += 1
                self.log(f"  request {req_id} ({len(docs)} docs) failed with "
                         f"{type(e).__name__}; splitting")
            self.submit(docs[:mid])
            self.submit(docs[mid:])
            return

        rows = [{
            "url":           d["url"],
            "is_legal":      "yes" if p >= YES_THRESHOLD else "no",
            "p_legal":       round(p, 4),
            "request":       req_id,
            "request_size":  len(docs),
            "input_tokens":  usage.input_tokens,
            "output_tokens": usage.output_tokens,
        } for d, p in zip(docs, probs)]
        n_yes = sum(r["is_legal"] == "yes" for r in rows)
        with self.lock:
            append_all(self.output, rows)
            s = self.stats
            s["yes"] += n_yes
            s["no"] += len(rows) - n_yes
            s["done"] += len(rows)
            s["requests"] += 1
            s["input_tokens"] += usage.input_tokens
            s["output_tokens"] += usage.output_tokens
            if s["done"] - self._last_report >= 50:
                self._last_report = s["done"]
                of = f"/{self.total}" if self.total else ""
                self.log(f"  {s['done']}{of} done  yes={s['yes']} no={s['no']} "
                         f"skipped={s['skipped']}  {s['requests']} requests  "
                         f"{time.time() - self.t0:.0f}s")

    def close(self):
        with self.idle:
            while self.inflight > 0:
                self.idle.wait(timeout=1.0)
        self.pool.shutdown(wait=True)
        if self.fatal:
            raise self.fatal
        s = self.stats
        s["seconds"] = time.time() - self.t0
        self.log(f"done: {s['yes'] + s['no']} new labels  yes={s['yes']} no={s['no']} "
                 f"skipped={s['skipped']}  {s['requests']} requests ({s['bisects']} bisects)  "
                 f"{s['input_tokens']:,} input tokens  {s['seconds']:.0f}s")
        return s


def run(client, pending, output, skipped, *, state_tokens=DEFAULT_STATE_TOKEN_BUDGET,
        workers=N_WORKERS, log=print):
    """Label every doc in `pending` ({url, text}, already truncated). The
    whole list is packed and submitted up front; see Labeler."""
    batches = list(pack(pending, state_tokens))
    log(f"{len(pending)} docs in {len(batches)} requests "
        f"(state budget {state_tokens} tokens, {workers} workers)")
    lab = Labeler(client, output, skipped, workers=workers, log=log, total=len(pending))
    for b in batches:
        lab.submit(b)
    return lab.close()


# --- main -----------------------------------------------------------------

def make_client():
    api_key = os.environ.get("JEV_API_KEY")
    if not api_key:
        sys.exit("JEV_API_KEY not set in .env")
    return TypeSafeClient(api_key=api_key, retry=RETRY)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input",  default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--state-tokens", type=int, default=DEFAULT_STATE_TOKEN_BUDGET,
                    help="estimated state tokens per request; 0 = one doc per request")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help="truncate page text to this many chars before sending")
    ap.add_argument("--workers", type=int, default=N_WORKERS,
                    help="concurrent requests")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many new docs (for a quick test)")
    args = ap.parse_args()

    skipped_path = os.path.splitext(args.output)[0] + ".skipped.jsonl"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    done = already_done(args.output)
    print(f"{len(done)} URLs already labeled in {args.output}, skipping those")

    # Sort out what actually needs querying before touching the API, so the
    # no-text rows are logged up front and the batches are all real docs.
    pending, n_no_text = [], 0
    for row in iter_jsonl(args.input):
        if row["url"] in done:
            continue
        if args.limit is not None and len(pending) >= args.limit:
            break
        if not row.get("text"):
            append_all(skipped_path, [{"url": row["url"], "reason": "no text"}])
            n_no_text += 1
            continue
        pending.append({"url": row["url"], "text": row["text"][:args.max_chars]})
    if n_no_text:
        print(f"{n_no_text} rows with no text written to {skipped_path}")

    with make_client() as client:
        stats = run(client, pending, args.output, skipped_path,
                    state_tokens=args.state_tokens, workers=args.workers)
    if stats["skipped"]:
        print(f"skipped rows in {skipped_path}")


if __name__ == "__main__":
    main()

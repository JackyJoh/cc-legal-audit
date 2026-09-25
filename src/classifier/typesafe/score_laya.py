"""
Scores pages with Laya, an open-weights System One model (ModernBERT-large,
421M) that runs locally, asking the same is-legal question Jev was asked.

Laya is the local candidate to replace or distill Jev. This is the zero-shot
baseline: the shipped checkpoint, no fine-tuning. Scores sit next to Jev's so
the two precision tables can be read side by side on the same hand labels.

The question. The definition goes into state, the way Jev received it, because
Laya caps the question plus its options at 192 tokens. It is a shortened copy
of the definition in prompts/legal_url_labeling_task.md: the LEGAL and
NON_LEGAL paragraphs verbatim, without the Common Crawl paragraph (its rule is
already in both) or the worked examples. The prompt file itself is unchanged,
since it is what the hand labels and Jev's scores were produced against.

Length. Laya's shipped checkpoint was trained at 512 tokens and truncates
state from the end. --max-len defaults to 4096, past what it was trained on,
so the page text is not cut to a few hundred tokens; whether its decisions
hold up at that length is part of what this measures. The definition sits
first in state so it is never the part that gets cut.

Text. Input rows may carry `text`; rows that don't are looked up by url in
--text-cache, the same extracted text Jev scored. Text is cut at 20,000 chars,
as for Jev. Rows with no text anywhere are counted and skipped.

Resumable: urls already in --output are skipped.

Runs in the laya environment (torch + laya), not the project venv:
  C:\\projects\\laya-env\\Scripts\\python.exe src/classifier/typesafe/score_laya.py
  ... --input a.jsonl b.jsonl --output out.jsonl --max-len 512
"""
import argparse
import json
import math
import os
import sys
import time

DEFAULT_INPUTS = [
    "data/candidates/jev_openweb_kept_full.jsonl",
    "data/candidates/jev_openweb_060_090_full.jsonl",
]
DEFAULT_OUTPUT = "data/candidates/laya_openweb_scores.jsonl"
DEFAULT_TEXT_CACHE = "data/candidates/_pool_text_cache.jsonl"
DEFAULT_JEV_SCORES = "data/labels/jev_openweb_scores.jsonl"
MODEL = "convaiinnovations/laya"
MAX_CHARS = 20_000                     # same cut as label_is_legal.DEFAULT_MAX_CHARS
PLACEHOLDERS = {None, "your_label"}
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

DEFINITION = (
    "LEGAL = the URL is from a source whose primary function is producing or "
    "publishing formal legal documents (court systems, legislative bodies, "
    "regulatory agencies, statute repositories, established legal publishers) "
    "AND the specific page's own HTML contains the actual text of a filing, "
    "statute, bill, regulation, or court opinion - not a page that merely links "
    "out to that text.\n\n"
    "NON_LEGAL (excluded even if legal-adjacent) = legal commentary, law firm "
    "marketing pages, legal news, homepage/search/index/menu pages of legal "
    "databases, and landing/index pages that merely link to the actual "
    "document text elsewhere - even on an otherwise-qualifying domain."
)

# Laya keeps at most ~90 tokens of instructions and 48 per option, so these stay short;
# the definition in state carries the detail.
QUESTIONS = {
    "is_legal": {
        "type": "noul",
        "instructions": "Apply `definition` to the page at `url` with content `text`. "
                        "Is this page a formal legal document as defined?",
        "criteria": {
            "true": "LEGAL: the page's own text is the primary source - a statute, bill, "
                    "regulation, filing, or court opinion - from a formal legal publisher.",
            "false": "NON_LEGAL: commentary, news, law firm marketing, or an index/landing/"
                     "homepage that only links to the document.",
        },
    }
}


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - half), min(1.0, c + half)


def load_rows(inputs):
    """Every input row, last line winning per url, in first-seen order."""
    rows = {}
    for path in inputs:
        for r in iter_jsonl(path):
            rows[r["url"]] = {**rows.get(r["url"], {}), **r}
    return list(rows.values())


def attach_text(rows, cache_path):
    """Fill `text` from the cache for rows that lack it. Streams the cache,
    holding only the urls asked for."""
    need = {r["url"] for r in rows if not r.get("text")}
    if need and os.path.exists(cache_path):
        found = {}
        for r in iter_jsonl(cache_path):
            if r["url"] in need and r.get("text"):
                found[r["url"]] = r["text"]
        for r in rows:
            if not r.get("text") and r["url"] in found:
                r["text"] = found[r["url"]]


def score(rows, output, max_len, batch_size):
    import laya                       # heavy; only needed when there is work to do

    agent = laya.load(MODEL)
    print(f"model {MODEL} on {agent.device}, max_len {max_len}, batch {batch_size}")
    t0, done = time.time(), 0
    chunk = batch_size * 8
    with open(output, "a", encoding="utf-8") as out:
        for i in range(0, len(rows), chunk):
            part = rows[i:i + chunk]
            states = [{"definition": DEFINITION, "url": r["url"], "text": r["text"][:MAX_CHARS]}
                      for r in part]
            results = agent.predict_batch(states, QUESTIONS, batch_size=batch_size,
                                          max_len=max_len, sort_by_length=len(part) > batch_size)
            for r, res in zip(part, results):
                out.write(json.dumps({"url": r["url"],
                                      "p_legal": round(res["answers"]["is_legal"]["noul"], 4),
                                      "label": r.get("label"),
                                      "model": MODEL, "max_len": max_len}) + "\n")
            out.flush()
            done += len(part)
            rate = done / (time.time() - t0)
            print(f"  {done:,}/{len(rows):,}  {rate:.1f} pages/s")


def precision_table(name, pairs):
    """pairs: (p, is_legal) for hand-labeled rows. Cumulative from the top."""
    print(f"\n{name}")
    print(f"{'t':>6}{'n':>7}{'legal':>8}{'precision':>11}{'wilson 95% CI':>22}")
    print("-" * 54)
    for t in THRESHOLDS:
        band = [lab for p, lab in pairs if p >= t]
        n, k = len(band), sum(band)
        if n == 0:
            print(f"{t:>6.2f}{0:>7}{'-':>8}{'-':>11}{'-':>22}")
            continue
        lo, hi = wilson(k, n)
        print(f"{t:>6.2f}{n:>7}{k:>8}{k / n:>11.4f}   [{lo:.4f}, {hi:.4f}]")


def report(output, jev_scores, max_len):
    laya_p = {r["url"]: (r["p_legal"], r.get("label")) for r in iter_jsonl(output)
              if r.get("max_len") == max_len}
    labeled = {u: lab == "legal" for u, (_, lab) in laya_p.items() if lab not in PLACEHOLDERS}
    if not labeled:
        print("\nno hand-labeled rows in the output, no precision table")
        return
    n_legal = sum(labeled.values())
    print(f"\nhand-labeled rows: {len(labeled)}  ({n_legal} legal, {len(labeled) - n_legal} non_legal)")

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else float("nan")

    print(f"median Laya p_legal: legal {med([laya_p[u][0] for u, l in labeled.items() if l]):.3f}"
          f"   non_legal {med([laya_p[u][0] for u, l in labeled.items() if not l]):.3f}")
    precision_table(f"Laya (max_len {max_len})", [(laya_p[u][0], l) for u, l in labeled.items()])

    if os.path.exists(jev_scores):
        jev = {r["url"]: r["p_legal"] for r in iter_jsonl(jev_scores)}
        both = [u for u in labeled if u in jev]
        if both:
            precision_table("Jev, same rows", [(jev[u], labeled[u]) for u in both])
    print("\nNote: rows drawn from Jev >= 0.60 only test separation inside that band, not what"
          " Laya would keep from pages Jev scored low.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--input", nargs="+", default=DEFAULT_INPUTS, help="one or more jsonl files with url (+ optional text, label)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--text-cache", default=DEFAULT_TEXT_CACHE, help="jsonl of {url, text} for rows without text")
    ap.add_argument("--jev-scores", default=DEFAULT_JEV_SCORES, help="Jev scores to print alongside; ignored if missing")
    ap.add_argument("--max-len", type=int, default=4096, help="token budget per page (Laya trained at 512)")
    ap.add_argument("--batch-size", type=int, default=8, help="pages per forward pass; lower if the GPU runs out of memory")
    args = ap.parse_args()

    rows = load_rows(args.input)
    done = set()
    if os.path.exists(args.output):
        done = {r["url"] for r in iter_jsonl(args.output) if r.get("max_len") == args.max_len}
    rows = [r for r in rows if r["url"] not in done]
    attach_text(rows, args.text_cache)
    no_text = [r for r in rows if not r.get("text")]
    rows = [r for r in rows if r.get("text")]
    print(f"{len(rows):,} to score, {len(done):,} already in {args.output}, {len(no_text)} with no text (skipped)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if rows:
        score(rows, args.output, args.max_len, args.batch_size)
    report(args.output, args.jev_scores, args.max_len)


if __name__ == "__main__":
    sys.exit(main())

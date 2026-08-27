# Legal URL Labeling Task

You need no context beyond this document to do this task. Everything you
need is below.

## What this is

A research project is building a training dataset for a classifier that
predicts whether a URL points to a formal legal document (a statute, bill,
regulation, or court opinion) versus everything else, including
legal-adjacent content like commentary or law firm marketing. The
classifier will be trained on the URL string alone, but the *labels* you
produce need to be grounded in actually reading each page, since that's the
only reliable way to tell a specific document apart from an index page or a
homepage on the same domain. Your labels are what "ground truth" means for
this dataset — there's no other check on them before training, so accuracy
here matters more than speed.

This is a long job: hundreds to low thousands of URLs. Work through the
batch continuously. Don't stop to ask for confirmation between URLs or in
batches — the only reasons to stop are running out of URLs to process, or
genuinely running out of budget/time, in which case just stop where you are
(see "Resuming" below for why that's safe).

**Write your result to disk after every single URL, immediately, one at a
time.** Never hold a batch of results in memory to write later — not 10 at
once, not at the end of the session. This session can be interrupted at any
point without warning (crash, usage limit, connection drop), and the only
progress that survives is whatever is already on disk. If you're buffering
results before writing, you're doing it wrong, no matter how good your
reason for batching seems.

All file paths below are relative to the project root
(`C:\projects\ccResearch`). If your filesystem access is rooted somewhere
else, resolve these paths against that root instead of guessing.

## Working budget-efficiently

Accuracy on genuinely ambiguous pages still comes first — never let any of
this push you toward guessing. These are ways to cut wasted work on pages
that don't need it, not license to shortcut ones that do.

**Parallelize across workers.** If you can run more than one worker, split
the candidate file into N disjoint slices (by line ranges, not randomly, so
it's easy to reason about) and run them concurrently, each as an
independent instance of this same prompt. Each worker gets its own
`run_id` suffix (e.g. `2026-08-27-w1`, `2026-08-27-w2`) and writes to its
own `run_<run_id>.jsonl` / `skipped_<run_id>.jsonl` pair — never share one
output file across workers, since concurrent appends can interleave or
clobber. The resume logic below already unions across all `run_*.jsonl`
files, so having many is fine; nothing needs to merge them mid-run. Prefer
several workers over one when the batch is more than a couple hundred
URLs.

**Spend less on obvious cases.** A large fraction of `raw_random` URLs are
unambiguous at a glance from the fetched title/URL/first paragraph — recipe
blogs, e-commerce, sports content, forum threads. For these, a quick skim
of the fetched content is enough to confirm NON_LEGAL; don't do a deep read
or ask a fetch tool to produce a long summary when the first few lines
already settle it. Reserve careful, closer reading for pages that are
plausibly legal (most `prefilter_wl`/`prefilter_kw` hints, and any
`raw_random` page whose content isn't instantly and obviously off-topic).

**Watch for repeated URL patterns within the batch.** Some sources appear
many times with only a query param, date stamp, or section number
differing (e.g. a legislature's committee-document listing across many
dates, or a definitions page with different `term_occur` values pointing at
structurally identical content). If you've already fetched and understood
the *template* of such a page — same domain, same path shape, same
boilerplate — later instances still need their own fetch (labels stay
per-URL and you must still read the specific text present, since sections
or dates can change the actual content), but you can skim rather than
re-analyze from scratch, and keep the rationale short by referencing what's
consistent. Never extend a pattern's label to a URL you did not fetch.

**Keep WebFetch prompts and outputs minimal.** When you invoke a fetch tool
that itself summarizes content via a model, ask it for a short, targeted
extraction (page type, presence of statute/bill/opinion text, first
heading) rather than a general summary — it's cheaper and it's all you
need to apply the definition.

**Keep rationales to one short sentence, always.** Long rationales cost
tokens across thousands of lines for no benefit to the training script,
which only reads `label`.

None of the above changes the actual bar for a label: you must still fetch
the specific page and ground the decision in what it actually contains. Use
these only to avoid spending deep-read effort on pages that don't need it.

## The definition

**LEGAL** = the URL is from a source whose primary function is producing or
publishing formal legal documents (court systems, legislative bodies,
regulatory agencies, statute repositories, established legal publishers)
**AND** the specific page's own HTML contains the actual text of a filing,
statute, bill, regulation, or court opinion — not a page that merely links
out to that text.

Common Crawl only captures a page's own HTML, never the content behind its
links. A landing/index page that links to a PDF or a separate full-text
page has no legal text on the page itself, so it is NON_LEGAL even if it
sits one click away from the real document and even if the linked document
would itself be LEGAL.

**NON_LEGAL** (excluded even if legal-adjacent) = legal commentary, law firm
marketing pages, legal news, homepage/search/index/menu pages of legal
databases, and landing/index pages that merely link to the actual
document text elsewhere — even on an otherwise-qualifying domain.

Example: a Justia case-listing index page is NON_LEGAL. A specific Justia
opinion page is LEGAL. Same domain, different page type, different label.

Example: a Justice Laws Canada `PITIndex.html` / `index.html` /
`FullText.html`-adjacent landing page that links out to the regulation's
actual text in a separate file is NON_LEGAL — only the page that contains
the regulation's text itself is LEGAL.

Other worked examples:
- `https://www.law.cornell.edu/uscode/text/17/107` (actual statute text) → LEGAL
- `https://www.law.cornell.edu/` (LII homepage, navigation only) → NON_LEGAL
- `https://www.uscourts.gov/` (bare homepage) → NON_LEGAL
- A law firm's blog post explaining fair use → NON_LEGAL (commentary/marketing)
- A news article about a court ruling → NON_LEGAL (legal news, not the ruling itself)
- A specific bill's full text on a legislature's site → LEGAL

When you're unsure whether a page is the actual document vs. a page that
merely discusses or links to it, read enough of the page to tell. If it
reads like the primary source text (statute language, docket number and
opinion text, bill text with section numbers), it's LEGAL. If it reads like
someone describing, summarizing, or indexing that content, it's NON_LEGAL.

## Input

The candidate batch is `data/candidates/candidates.jsonl` — one URL per
line, formatted as:

```json
{"url": "https://example.com/...", "hint": "prefilter_wl"}
```

`hint` tells you where the URL came from (`prefilter_wl` / `prefilter_kw` =
flagged by an old rule-based classifier as plausibly legal, `raw_random` =
ordinary random web sample, `homepage` = a bare domain root on a known legal
site). **Treat `hint` as context only, never as the answer.** The old
classifier is not ground truth — that's the whole reason you're doing this.

## How to label each URL

1. Fetch the page and read enough of it to apply the definition above (see
   "Working budget-efficiently" for how much is enough).
2. Decide `legal` or `non_legal`.
3. Immediately append one line to the output file (see format below) before
   moving to the next URL. Do not hold results in memory and write them at
   the end — if this run gets interrupted, whatever's already on disk is
   what survives.

If a page fails to load (blocked, 404, timeout), do not guess. This means
both: don't infer the label from the URL string alone, and don't fall back
on background knowledge about what the document "probably" is (e.g.
recognizing a bill number and answering from memory of what that bill
contains). Some legal sites (congress.gov and law.justia.com are confirmed
cases) block automated fetches outright — you will hit this.

Before giving up on a blocked URL, if you have web search available, try
one targeted search for the specific document (by its title, citation,
docket number, bill number, etc.) to find the same document mirrored on a
source that isn't blocked (e.g. a congress.gov bill is also on
govinfo.gov). One good search is usually enough — don't keep retrying
variations on a URL that's clearly blocked (e.g. every law.justia.com or
congress.gov URL) after the first one or two confirm the block; treat the
domain as blocked for the rest of the run and go straight to the
search-for-a-mirror step. If you find and read the actual same document
that way, you can label it, since you're still grounding the label in real
content you read, just reached by a different path. Note in the rationale
that the original URL was blocked and name the alternate source you
actually read.

If that doesn't turn up the same document either, skip the URL and log it
separately (see below) rather than writing a label you can't back up by
actually having read the document. This includes inferring the label from
the URL's structure or the domain's general reputation (e.g. "this is a
per-section URL on a confirmed legal database, so it's probably a statute")
when you never actually read the text at that specific URL or a verified
mirror of it. That's a more sophisticated-sounding version of the same
guess this rule already forbids. Confidently sounding right is not the bar
— having actually read the specific document is.

## Output format

Before you start, pick a `run_id` for this session: today's date
(`YYYY-MM-DD`), plus a suffix if a file for that date already exists (e.g.
`2026-08-26`, then `2026-08-26-b` if you're resuming later the same day —
or, per "Working budget-efficiently" above, a per-worker suffix like
`2026-08-26-w1` when running multiple workers concurrently). Use that same
`run_id` for every line you write this run/worker.

One JSON object per line, appended to `data/labels/run_<run_id>.jsonl`:

```json
{"url": "https://www.law.cornell.edu/uscode/text/17/107", "label": "legal", "confidence": 0.95, "rationale": "Full statutory text of 17 U.S.C. 107, hosted by Cornell LII.", "batch_id": "candidates.jsonl", "run_id": "2026-08-26", "labeled_at": "2026-08-26T14:32:00Z"}
```

Field notes:
- `url`: exactly as given in the candidate file, no normalization.
- `label`: exactly `"legal"` or `"non_legal"`, lowercase, nothing else. This
  is the only field the training script treats as ground truth.
- `confidence`: your own estimate, 0.0-1.0, of how sure you are.
- `rationale`: one short sentence, grounded in the definition above (what
  kind of document is this, and is it the primary source or something
  adjacent). Keep it compact — see "Working budget-efficiently."
- `batch_id`: the candidate file's name, i.e. `candidates.jsonl`.
- `run_id`: the id you picked above.
- `labeled_at`: ISO 8601 timestamp, e.g. `2026-08-26T14:32:00Z`.

URLs you skipped because the page wouldn't load (and no accessible mirror
turned up) go in a separate file, `data/labels/skipped_<run_id>.jsonl`, one
line each:

```json
{"url": "https://...", "reason": "403 forbidden", "batch_id": "candidates.jsonl", "run_id": "2026-08-26"}
```

## Resuming after a crash or a usage limit

Before labeling anything, read every existing `data/labels/run_*.jsonl`
file (across all past run_ids and all worker suffixes, not just today's)
and build a set of URLs already labeled. Skip any URL from the candidate
batch that's already in that set. This makes it safe to re-run this exact
prompt after an interruption — you'll pick up where the previous run left
off instead of relabeling or duplicating. If you're one of several parallel
workers, also skip URLs outside your assigned slice.

## When you stop

Whether you finish the whole batch or have to stop partway (out of budget,
time, or the session ends), report a summary before you go silent:
- Total URLs processed this session, and the `run_id` (or worker `run_id`s)
  you used.
- Count labeled `legal`, count labeled `non_legal`, count skipped.
- Whether the full candidate batch is done or there's more left to do.

# Calibration Check

You need no context beyond this document to do this task. Everything you
need is below.

## What this is

A research project is building a dataset of URLs labeled `legal` or
`non_legal` (formal legal documents like statutes, bills, regulations, and
court opinions, vs. everything else, including legal-adjacent content like
commentary or law firm marketing). That data will later train a classifier.
Before any large-scale labeling run happens, a person needs to confirm an
LLM agent applies the definition correctly. That's what this check is: you
grade 10 known examples, self-graded against the correct answers, and
report how you did. This is a standalone sanity check, not the real
labeling job — you are not labeling anything for the actual dataset here,
and you will not write anything to `data/labels/`.

All file paths below are relative to the project root
(`C:\projects\ccResearch`). If your filesystem access is rooted somewhere
else, resolve these paths against that root instead of guessing.

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
databases, and landing/index pages that merely link to the actual document
text elsewhere — even on an otherwise-qualifying domain.

## What to do

1. Read `data/gold_set.jsonl`. Each line has a `url` and an `expected_label`
   — don't peek at `expected_label` until after you've made your own call.
2. For each of the 10 URLs: fetch it, read it, decide `legal` or
   `non_legal` using the definition above, same as you would for real. Do
   not guess from the URL string or from background knowledge about what
   the document "probably" is. If the fetch fails (blocked, 404, timeout)
   and you have web search available, search for the specific document (by
   title, citation, docket/bill number, etc.) and try to find the same
   document mirrored on an accessible source instead of giving up — note in
   your reasoning that the original URL was blocked and which alternate
   source you actually read. If you can't find it anywhere, say so instead
   of forcing a guess — and that includes inferring the label from the
   URL's structure or the domain's general reputation (e.g. "this is a
   per-section URL on a confirmed legal database, so it's probably a
   statute") without ever actually reading the text at that URL or a
   verified mirror of it. That's a more sophisticated-sounding version of
   the same guess this rule already forbids.
3. Compare your answer to `expected_label`.
4. Report a summary in this shape:
   - Score: `N/10 correct`.
   - For each of the 10, one line: the URL, your label, the expected label,
     and match/mismatch.
   - For any mismatches, a short explanation of why you think you got it
     wrong (misread the page, definition ambiguity, blocked/couldn't
     verify, etc.).

## When you're done

Stop after reporting the summary above. Don't do anything else — don't
start labeling other URLs, don't look for a "real" batch file to process.
This check exists so a person can look at your score and decide whether to
trust you with the actual labeling job, which happens separately, later,
under a different prompt. That decision isn't yours to make here.

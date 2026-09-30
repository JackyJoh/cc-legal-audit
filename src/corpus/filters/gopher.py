"""
Gopher's document quality filters: decides whether a page is clean enough to
keep, using the 21 rules from the Gopher paper (appendix A.1.1).

Each page is judged only against itself, and is either kept whole or dropped
whole; comparing pages to each other is deduplication, a later step. The rules
run in two stages, in the paper's order, and a page is dropped at the first
rule it fails:

  quality     8 rules on the page's basic shape: length, word length, symbols,
              bullets, letters, and whether it reads like English
  repetition  13 rules on how much the page repeats its own lines, paragraphs
              and phrases

Paper, not library. The thresholds are the paper's. Where the paper leaves a
detail open (what a "word" or a "bullet" is), the choice is marked "open:" in a
comment. The one that matters most: a word is a whitespace-separated token, so
punctuation stays attached to its word. Libraries that split punctuation into
separate words (datatrove, used by FineWeb) drop far more citation-heavy legal
text on the letters rule.

Usage:
  sys.path.insert(0, "src/corpus/filters")
  from gopher import gopher_reject
  reason = gopher_reject(page_text)   # None = keep, else the failed rule's name

Outputs: none. Not run directly; imported.
"""
import re
from collections import Counter

# stage 1: quality
MIN_WORDS, MAX_WORDS         = 50, 100_000
MIN_MEAN_WORD, MAX_MEAN_WORD = 3, 10
MAX_HASH_RATIO               = 0.1     # '#' per word
MAX_ELLIPSIS_RATIO           = 0.1     # ellipses per word
MAX_BULLET_LINES             = 0.90    # share of lines starting with a bullet
MAX_ELLIPSIS_LINES           = 0.30    # share of lines ending with an ellipsis
MIN_ALPHA_WORDS              = 0.80    # share of words with at least one letter
MIN_STOP_WORDS               = 2       # different words from STOP_WORDS
STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}

# stage 2: repetition (the paper's Table A1)
MAX_DUP_LINES      = 0.30    # share of lines that repeat an earlier line
MAX_DUP_PARAS      = 0.30    # share of paragraphs that repeat an earlier one
MAX_DUP_LINE_CHARS = 0.20    # share of the page's characters in those lines
MAX_DUP_PARA_CHARS = 0.20    # share of the page's characters in those paragraphs
TOP_NGRAM_MAX = {2: 0.20, 3: 0.18, 4: 0.16}
DUP_NGRAM_MAX = {5: 0.15, 6: 0.14, 7: 0.13, 8: 0.12, 9: 0.11, 10: 0.10}

# open: the paper names neither the ellipsis nor the bullet characters
ELLIPSES = ("...", "…")
BULLETS  = ("•", "●", "○", "▪", "‣", "-", "*", "–")
EDGE_PUNCT = re.compile(r"^\W+|\W+$")   # non-word characters at either end of a word


# first rule the page fails, or None if it passes every rule
def gopher_reject(text):
    return quality_reject(text) or repetition_reject(text)


# stage 1: the 8 quality rules, in the paper's order
def quality_reject(text):
    words = text.split()          # open: a word is a whitespace-separated token
    n = len(words)
    if not MIN_WORDS <= n <= MAX_WORDS:
        return "word count"
    if not MIN_MEAN_WORD <= sum(map(len, words)) / n <= MAX_MEAN_WORD:
        return "mean word length"
    if text.count("#") / n > MAX_HASH_RATIO:
        return "hash ratio"
    if sum(text.count(e) for e in ELLIPSES) / n > MAX_ELLIPSIS_RATIO:
        return "ellipsis ratio"

    lines = non_empty(text.splitlines())
    if sum(l.startswith(BULLETS) for l in lines) / len(lines) > MAX_BULLET_LINES:
        return "bullet lines"
    if sum(l.endswith(ELLIPSES) for l in lines) / len(lines) > MAX_ELLIPSIS_LINES:
        return "ellipsis lines"
    if sum(any(c.isalpha() for c in w) for w in words) / n < MIN_ALPHA_WORDS:
        return "alphabetic words"

    # open: matched case-insensitively with surrounding punctuation stripped
    # (curly quotes included), so "The", "that," and "“the" all count
    found = {EDGE_PUNCT.sub("", w).lower() for w in words} & STOP_WORDS
    if len(found) < MIN_STOP_WORDS:
        return "stop words"
    return None


# stage 2: the 13 repetition rules, in Table A1's order
def repetition_reject(text):
    total = len(text)
    lines = non_empty(text.splitlines())
    paras = non_empty(re.split(r"\n\s*\n", text))   # open: paragraphs split on blank lines
    if not lines:
        return None                                 # blank page: nothing repeats
    dup_lines, dup_line_chars = duplicates(lines)
    dup_paras, dup_para_chars = duplicates(paras)

    if dup_lines / len(lines) > MAX_DUP_LINES:
        return "duplicate lines"
    if dup_paras / len(paras) > MAX_DUP_PARAS:
        return "duplicate paragraphs"
    if dup_line_chars / total > MAX_DUP_LINE_CHARS:
        return "duplicate line chars"
    if dup_para_chars / total > MAX_DUP_PARA_CHARS:
        return "duplicate paragraph chars"

    words = text.split()
    for n, cap in TOP_NGRAM_MAX.items():
        if top_ngram_chars(words, n) / total > cap:
            return f"top {n}-gram"
    for n, cap in DUP_NGRAM_MAX.items():
        if dup_ngram_chars(words, n) / total > cap:
            return f"duplicate {n}-grams"
    return None


# stripped pieces with the blank ones removed
def non_empty(pieces):
    return [p.strip() for p in pieces if p.strip()]


# how many pieces repeat an earlier piece, and how many characters they hold
def duplicates(pieces):
    seen, count, chars = set(), 0, 0
    for p in pieces:
        if p in seen:
            count += 1
            chars += len(p)
        seen.add(p)
    return count, chars


# characters covered by the most frequent n-gram: its length times its count
def top_ngram_chars(words, n):
    grams = Counter(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    if not grams:
        return 0
    gram, count = grams.most_common(1)[0]
    return len(gram) * count


# characters in n-grams that repeat an earlier n-gram; after a repeat the scan
# jumps past it, so overlapping repeats are counted once
def dup_ngram_chars(words, n):
    seen, chars, i = set(), 0, 0
    while i <= len(words) - n:
        gram = " ".join(words[i:i + n])
        if gram in seen:
            chars += len(gram)
            i += n
        else:
            seen.add(gram)
            i += 1
    return chars

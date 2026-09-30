"""
C4's page-level content checks: decides whether a page is placeholder text or
code, using two of the rules from the C4 paper (Raffel et al. 2020, section
2.2).

Each page is judged only against itself, and is either kept whole or dropped
whole. A page is dropped at the first rule it fails, in C4's order:

  lorem ipsum     "lorem ipsum" appears anywhere (any case): a template page
  curly bracket   "{" appears anywhere: the page probably contains code

C4's other rules are not here. Its line rules (terminal punctuation, short
lines, "javascript", policy notices, citation markers) edit page text; its
bad-words, sentence-count and language rules were left out on purpose.

Paper, not library. C4's own code, and datatrove's C4QualityFilter, run these
two checks only on the lines that survive C4's line rules. With no line rules
here, they run on the whole page, as the paper states them, so they can drop
slightly more than C4 did.

Usage:
  sys.path.insert(0, "src/corpus/filters")
  from c4 import c4_reject
  reason = c4_reject(page_text)   # None = keep, else the failed rule's name

Outputs: none. Not run directly; imported.
"""


# first rule the page fails, or None if it passes both
def c4_reject(text):
    if "lorem ipsum" in text.lower():
        return "lorem ipsum"
    # open: the paper says "a curly bracket"; C4's code and datatrove check "{" only
    if "{" in text:
        return "curly bracket"
    return None
"""
The full quality-filter pass: decides whether a page is kept, by running every
filter in order and stopping at the first rule the page fails.

Each page is judged only against itself, and is either kept whole or dropped
whole; no filter edits page text. The order:

  url     url.py      adult-site blocklist and banned URL words (URL only)
  gopher  gopher.py   Gopher's 8 quality and 13 repetition rules
  c4      c4.py       C4's lorem ipsum and curly bracket checks

The URL filter runs first because it needs no page text, so a page it drops
never has to be fetched.

Usage:
  sys.path.insert(0, "src/corpus/filters")
  from pipeline import page_reject
  reason = page_reject(page_url, page_text)   # None = keep, else "filter: rule"

Outputs: none. Not run directly; imported.
"""
from c4 import c4_reject
from gopher import gopher_reject
from url import url_reject

# first rule the page fails as "filter: rule" (e.g. "gopher: word count"), or None to keep
def page_reject(url, text):
    if reason := url_reject(url):
        return f"url: {reason}"
    if reason := gopher_reject(text):
        return f"gopher: {reason}"
    if reason := c4_reject(text):
        return f"c4: {reason}"
    return None
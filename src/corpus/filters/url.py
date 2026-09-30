"""
URL filter: decides whether a page comes from an adult site, using datatrove's
URLFilter, the URL filter in Hugging Face's FineWeb pipeline.

Only the URL is read, never the page text, and a page is kept or dropped
whole. A page is dropped at the first check it fails:

  blocked domain         its domain is on the blocklist of adult sites (built
                         from the UT1 blacklist, Universite Toulouse 1
                         Capitole's list of websites by category)
  blocked subdomain      its full hostname is on that blocklist
  blocked url            the exact URL is on the blocklist
  banned url word        the URL, split on anything not a letter or digit,
                         contains one hard-banned word
  soft banned url words  ... or 2 or more soft-banned words
  banned url substring   a banned string appears anywhere in the URL, which
                         catches words run together (freeporn...)

Library, not paper. Unlike gopher.py, this runs datatrove's code and lists
unchanged, so it drops what FineWeb's URL filter drops. The lists are not
adjusted for legal text: a URL like .../sex-offenses/ can be dropped, as it
would be in a real pipeline.

Needs datatrove and pyahocorasick. The first call downloads the domain and URL
blocklists from the Hugging Face hub; later calls use the cached copy.

Usage:
  sys.path.insert(0, "src/corpus/filters")
  from url import url_reject
  reason = url_reject(page_url)   # None = keep, else the failed check's name

Outputs: none. Not run directly; imported.
"""
from datatrove.data import Document
from datatrove.pipeline.filters import URLFilter

# datatrove's reason codes, renamed to read like gopher.py's
REASONS = {
    "domain":              "blocked domain",
    "subdomain":           "blocked subdomain",
    "url":                 "blocked url",
    "hard_blacklisted":    "banned url word",
    "soft_blacklisted":    "soft banned url words",
    "blacklisted_subword": "banned url substring",
}

_filter = None   # built on first use, so importing this file downloads nothing


# first check the URL fails, or None if it passes every check
def url_reject(url):
    global _filter
    if _filter is None:
        _filter = URLFilter()     # datatrove's defaults: integrated lists, soft-word threshold 2
    result = _filter.filter(Document(text="", id=url, metadata={"url": url}))
    return None if result is True else REASONS[result[1]]
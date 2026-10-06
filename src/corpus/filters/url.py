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

Shared domain list. datatrove keeps its 4.5M blocked domains as a Python set,
about 660 MB in every process that loads it. Here the same domains are stored
once on disk as sorted 64-bit hashes (about 36 MB, next to datatrove's cached
lists), built by a short-lived helper process, and every process opens that
one file read-only, so the operating system shares a single copy. Lookups
give the same answers as datatrove's set; every other list is loaded exactly
as datatrove loads it.

Needs datatrove (pinned) and pyahocorasick. The first call downloads the
blocklists from the Hugging Face hub and builds the hash file; later calls use
both cached copies.

Usage:
  sys.path.insert(0, "src/corpus/filters")
  from url import url_reject
  reason = url_reject(page_url)   # None = keep, else the failed check's name

Outputs: none. Not run directly; imported.
"""
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from datatrove.data import Document
from datatrove.pipeline.filters import URLFilter
from datatrove.pipeline.filters.url_filter import (ASSETS_PATH, cached_assets_path, get_list,
                                                   safely_create_file)

# datatrove's reason codes, renamed to read like gopher.py's
REASONS = {
    "domain":              "blocked domain",
    "subdomain":           "blocked subdomain",
    "url":                 "blocked url",
    "hard_blacklisted":    "banned url word",
    "soft_blacklisted":    "soft banned url words",
    "blacklisted_subword": "banned url substring",
}

LISTS   = "url_filterblacklistsv0_3_0"        # datatrove 0.10.1's blocklist archive
CACHE   = cached_assets_path(library_name="datatrove", namespace="filters", subfolder="url_filter")
DOMAINS = os.path.join(CACHE, f"{LISTS}_domains.u64.npy")

_filter = None   # built on first use, so importing this file downloads nothing


def hash64(text):
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


# the blocked domains as a sorted array of hashes, opened read-only so every
# process shares one copy; `in` works like datatrove's set
class HashedDomains:
    def __init__(self, path):
        self.hashes = np.load(path, mmap_mode="r")

    def __contains__(self, domain):
        h = np.uint64(hash64(domain))
        i = np.searchsorted(self.hashes, h)
        return i < len(self.hashes) and self.hashes[i] == h


# runs in the helper process: datatrove's own domain list, hashed, sorted and saved
def build_domains():
    full = URLFilter()
    full.download_data()
    hashes = np.unique(np.fromiter((hash64(d) for d in full.block_listed_domains),
                                   dtype=np.uint64, count=len(full.block_listed_domains)))
    np.save(DOMAINS + ".tmp.npy", hashes)
    os.replace(DOMAINS + ".tmp.npy", DOMAINS)


# build the hash file in a helper process if it's missing; the helper's 660 MB
# goes back to the system when it exits
def prepare():
    if not os.path.exists(DOMAINS):
        with ProcessPoolExecutor(1) as ex:
            ex.submit(build_domains).result()


# datatrove's URLFilter, loading the domain list from the shared hash file
class SharedURLFilter(URLFilter):
    # datatrove 0.10.1's download_data, with only the domain list replaced
    def download_data(self):
        if self._downloaded:
            return
        prepare()
        file_to_lock = os.path.join(CACHE, f"{LISTS}.tar.gz")

        def do_extract():
            import tarfile
            with tarfile.open(os.path.join(ASSETS_PATH, f"{LISTS}.tar.gz"), "r:gz") as tar:
                tar.extractall(CACHE)

        safely_create_file(file_to_lock, do_extract)
        self.block_listed_domains = HashedDomains(DOMAINS)
        self.block_listed_url = get_list(CACHE, "urls", self.block_listed_url, do_normalize=False)
        self.banned_words = get_list(ASSETS_PATH, "banned_words.txt", self.banned_words)
        self.banned_subwords = get_list(ASSETS_PATH, "banned_subwords.txt", self.banned_subwords)
        self.soft_banned_words = get_list(ASSETS_PATH, "soft_banned_words.txt", self.soft_banned_words)
        for word in self.banned_subwords:
            self.banned_subwords_automaton.add_word(word, len(self.banned_subwords_automaton))
        self.banned_subwords_automaton.make_automaton()
        self._downloaded = True


# first check the URL fails, or None if it passes every check
def url_reject(url):
    global _filter
    if _filter is None:
        _filter = SharedURLFilter()   # datatrove's defaults: integrated lists, soft-word threshold 2
    result = _filter.filter(Document(text="", id=url, metadata={"url": url}))
    return None if result is True else REASONS[result[1]]
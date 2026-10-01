"""
MinHash building blocks: turns a page into the numbers near-duplicate
detection compares, and measures how similar two pages really are.

Similarity is Jaccard over word 13-grams, as in Gopher: each page becomes the
set of every 13-word phrase in it, and two pages' similarity is the share of
phrases they have in common (1.0 = identical). The pieces:

  shingle_hashes   a page's 13-word phrases, each hashed to a 64-bit number
  signature        128 numbers summarizing those phrases (MinHash); two pages
                   agree on any one of them with probability equal to their
                   Jaccard
  band_keys        the signature cut into 32 bands of 4, each band hashed to
                   one key (LSH); pages sharing a key in any band become
                   candidate pairs
  estimate         Jaccard estimated from two signatures
  exact_jaccard    Jaccard computed exactly from two pages' phrase hashes

And the shared pieces every dedup step uses:

  page_id          a page's ID: its URL hashed to 63 bits, so it fits the
                   signed 64-bit integers numpy, DuckDB and pandas use
  text_hash        one number for a page's words, ignoring case, spacing and
                   punctuation: identical text, identical number
  survivor_rank    a fixed random number per page ID; in any group of copies,
                   the page with the lowest one is kept

32 bands of 4 put LSH's cutoff near 0.42: a pair at true Jaccard 0.6 shares a
band about 99% of the time, and every candidate is then checked exactly.

Same as the pilot. Words, phrases, hashes and permutations are the ones in
minhash_pilot/minhash_pairs.py (same seed), so a page gets the same signature
in both.

Usage:
  sys.path.insert(0, "src/corpus/dedup")
  from minhash import shingle_hashes, signature, band_keys, exact_jaccard
  sh = shingle_hashes(page_text)
  keys = band_keys(signature(sh))   # 32 keys, one per band

Outputs: none. Not run directly; imported.
"""
import hashlib
import re

import numpy as np

NGRAM     = 13
NUM_PERM  = 128
NUM_BANDS = 32
ROWS      = NUM_PERM // NUM_BANDS    # 4 signature numbers per band
SEED      = 42

# open: Gopher ignores whitespace and punctuation but doesn't define a word;
# here a word is a run of letters, digits or underscores, lowercased
WORD = re.compile(r"\w+")

# permutation h(x) = (a*x + b) mod P over 32-bit shingle hashes; a, b < 2^31
# keeps a*x + b inside uint64
P = np.uint64(4_294_967_311)                     # prime just above 2^32
_rng = np.random.default_rng(SEED)
A = _rng.integers(1, 2**31, NUM_PERM, dtype=np.uint64)
B = _rng.integers(0, 2**31, NUM_PERM, dtype=np.uint64)


# sorted unique 64-bit hashes of the page's word 13-grams
def shingle_hashes(text):
    words = WORD.findall(text.lower())
    if len(words) < NGRAM:
        grams = {" ".join(words)}                # short page: one shingle
    else:
        grams = {" ".join(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}
    return np.unique(np.array([hash64(g.encode()) for g in grams], dtype=np.uint64))


# MinHash signature: the minimum of each of the NUM_PERM permutations over the shingles
def signature(hashes):
    x = hashes & np.uint64(0xFFFFFFFF)
    return ((A[:, None] * x[None, :] + B[:, None]) % P).min(axis=1)


# LSH keys: one 64-bit hash per band of ROWS signature numbers; keys only
# match within the same band, so find pairs by (band index, key)
def band_keys(sig):
    return np.array([hash64(band.tobytes()) for band in sig.reshape(NUM_BANDS, ROWS)],
                    dtype=np.uint64)


# Jaccard estimated from two signatures: the share of positions that agree
def estimate(sig_a, sig_b):
    return float((sig_a == sig_b).mean())


# true Jaccard of two sorted unique shingle-hash arrays
def exact_jaccard(a, b):
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


# 63-bit hash of the URL
def page_id(url):
    return hash64(url.encode()) >> 1


# hash of the page's words joined by single spaces (the same words shingle_hashes uses)
def text_hash(text):
    return hash64(" ".join(WORD.findall(text.lower())).encode())


# fixed random number per page ID (splitmix64 of id ^ SEED); works on arrays
def survivor_rank(ids):
    z = np.asarray(ids).astype(np.uint64) ^ np.uint64(SEED)
    with np.errstate(over="ignore"):              # wrapping multiplication is the point
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash64(data):
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")
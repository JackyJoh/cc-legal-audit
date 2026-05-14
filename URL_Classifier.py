import os
import re
from urllib.parse import urlparse


WHITELIST_FILE = os.path.join(os.path.dirname(__file__), "wl_candidates.txt")

suffixAllowed = 5  # keywords shorter than this require exact token match, not suffix

# Keywords whose presence in a URL hostname reliably signals a primary legal source.
# Only terms that survive host-only tokenization with low false-positive rates.
legalKeywords = [
    'judicial', 'statute', 'litigation', 'counsel',
    'judiciary', 'verdict', 'plaintiff', 'defendant', 'indictment',
    'jurisprudence', 'justice', 'tribunal', 'appeal', 'senate',
]


def LoadWhitelist(path):
    domains = set()
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                domain = parts[1]
                # Strip www. so one entry covers both www.uscourts.gov and cand.uscourts.gov
                if domain.startswith('www.'):
                    domain = domain[4:]
                domains.add(domain)
    except FileNotFoundError:
        pass
    return domains


WHITELIST = LoadWhitelist(WHITELIST_FILE)


def InWhitelist(host):
    # Strip www. from host before matching so suffix check catches all subdomains
    if host.startswith('www.'):
        host = host[4:]
    return any(host == entry or host.endswith('.' + entry) for entry in WHITELIST)


def classify(url: str) -> bool:
    """Return True if the URL's host is a primary legal source.

    Checks the curated domain whitelist first, then falls back to keyword
    matching on the hostname only. Path components are ignored to prevent
    false positives from /case-studies/ or /legal pages.
    """
    host = urlparse(url).netloc

    if InWhitelist(host):
        return True

    tokens = re.split(r'[.,-/=?_]', host)
    for t in tokens:
        candidates = [t]
        if t.endswith('es'):
            candidates.append(t[:-2])
        elif t.endswith('s'):
            candidates.append(t[:-1])

        for word in legalKeywords:
            for c in candidates:
                if len(word) >= suffixAllowed and c.endswith(word):
                    return True
                if c == word:
                    return True

    return False


if __name__ == "__main__":
    print(classify("https://www.uscourts.gov/court-locator"))
    print(classify("https://www.amazon.com/products/shoes"))
    print(classify("https://www.flsenate.gov/laws/statutes/2020/561.42"))
    print(classify("https://law.cornell.edu/uscode/text/17/107"))
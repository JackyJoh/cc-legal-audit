import os
import re
from urllib.parse import urlparse


WHITELIST_FILE = os.path.join(os.path.dirname(__file__), "wl_candidates.txt")

# Keywords whose presence in a URL hostname reliably signals a primary legal source.
# Only terms that survive host-only tokenization with low false-positive rates.
legalKeywords = [
    'judicial', 'statute', 'litigation', 'counsel',
    'judiciary', 'plaintiff', 'defendant', 'indictment',
    'jurisprudence', 'tribunal',
]

# Keyword fallback is restricted to non-commercial/non-academic TLDs.
# .com/.edu/.org domains with legal keywords are almost always law firms,
# university governance bodies, or advocacy orgs — not primary legal sources.
_KW_BLOCKED_TLDS = ('.com', '.edu', '.org', '.net')


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
    """Return True if the URL is a primary legal source document.

    Filters homepage and search paths first, then checks the curated domain
    whitelist, then falls back to keyword matching on the hostname only.
    """
    parsed = urlparse(url)
    path = parsed.path.lower().rstrip('/')
    if not path or path.startswith('/search'):
        return False

    host = parsed.netloc.split(':', 1)[0]

    if InWhitelist(host):
        return True

    if host.endswith(_KW_BLOCKED_TLDS):
        return False

    tokens = re.split(r'[.,-/=?_]', host)
    for t in tokens:
        candidates = [t]
        if t.endswith('es'):
            candidates.append(t[:-2])
        elif t.endswith('s'):
            candidates.append(t[:-1])

        for word in legalKeywords:
            if word in candidates:
                return True

    return False


if __name__ == "__main__":
    print(classify("https://www.uscourts.gov/court-locator"))
    print(classify("https://www.amazon.com/products/shoes"))
    print(classify("https://www.flsenate.gov/laws/statutes/2020/561.42"))
    print(classify("https://law.cornell.edu/uscode/text/17/107"))
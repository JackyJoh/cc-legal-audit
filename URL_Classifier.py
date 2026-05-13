import re


# Minimum keyword length to permit suffix matching. Short keywords (e.g. 'irs', 'epa')
# match exact tokens only to avoid false positives like 'theirs' matching 'irs'.
suffixAllowed = 5

# Legal domain keywords derived from Pile of Law subcorpus labels and CC frequency scan.
legalKeywords = ['law', 'legal', 'court', 'judicial', 'statute', 'attorney',
                  'lawyer', 'litigation', 'counsel', 'judiciary', 'verdict',
                  'plaintiff', 'defendant', 'jurisprudence', 'case',
                  'justice', 'tribunal', 'appeal', 'filing']

def classify(url: str) -> bool:
    """Classify a URL as legal or non-legal.

    Tokenizes the URL on common delimiters and checks each token against
    legalKeywords. Tokens are checked as-is and with trailing 's'/'es' stripped.
    Keywords >= suffixAllowed characters also match as a suffix of a token.

    Args:
        url: The full URL string to classify.

    Returns:
        True if the URL contains a legal keyword token, False otherwise.
    """
    tokens = re.split(r'[.,-/=?_]', url)

    for t in tokens:
        candidates = [t]
        if t.endswith('es'): candidates.append(t[:-2])
        elif t.endswith('s'): candidates.append(t[:-1])

        for word in legalKeywords:
            for c in candidates:
                if len(word) >= suffixAllowed:
                    if c.endswith(word): return True
                if c == word: return True
    return False


if __name__ == "__main__":
    print(classify("https://www.uscourts.gov/court-locator"))
    print(classify("https://www.amazon.com/products/shoes"))
    print(classify("https://www.flsenate.gov/laws/statutes/2020/561.42"))
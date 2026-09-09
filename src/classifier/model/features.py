"""
Feature recipes and label loading for both classifiers.

Two modes: "url" vectorizes the URL string (char 3-5 grams), "text"
vectorizes the extracted page body (word 1-2 grams). Everything else about
the two models is identical.

Train, score and eval all import from here so they cannot drift onto
different settings, which would make their numbers incomparable without
anything visibly breaking.

Also holds the domain purity filter. TF-IDF on page text picks up site
template text (nav bars, footers) as readily as real content, so after
vectorizing, this counts how many distinct registered domains each n-gram
appears on and drops the ones below MIN_DOMAIN_DF. Boilerplate lives on one
site; real legal vocabulary appears across many. The count is taken on the
training fold only, so a held-out publisher never influences the feature set.
"""
import json
import os
from collections import Counter
from urllib.parse import urlparse

import joblib
import numpy as np
import tldextract
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELED_URLS = "data/processed/labeled_urls.jsonl"
LABELED_TEXT = "data/processed/labeled_text.jsonl"

# no network lookup: the bundled suffix list is deterministic across runs and
# machines, which a refreshed one would not be
_extract = tldextract.TLDExtract(suffix_list_urls=())

# Drop any n-gram confined to fewer than this many registered domains. Set
# from the observed gap in domain-DF rather than tuned against a score:
# Justice Canada's page template sits at 1-2 domains ('marginal note' 1,
# 'details date' 1, 'date modified' 2 across justice.gc.ca and crtc.gc.ca),
# while genuine register vocabulary starts at 17 ('general assembly') and
# runs to 79 ('section'). 3 is the smallest value that clears all four
# template phrases; 4 drops vocabulary retention off a cliff, 72% -> 49%.
MIN_DOMAIN_DF = 3


class DomainPurityTfidf:
    """TF-IDF that drops features appearing on fewer than min_domain_df
    distinct registered domains.

    Pass the training fold's domains and nothing else. Counting over the full
    dataset lets a held-out publisher decide which features exist, which is
    the leak the grouped and LODO splits are there to detect.
    """

    def __init__(self, min_domain_df=1, **kwargs):
        self.min_domain_df = min_domain_df
        self.vec = TfidfVectorizer(**kwargs)
        self.keep_ = None

    def fit_transform(self, docs, domains=None):
        X = self.vec.fit_transform(docs)
        if self.min_domain_df <= 1 or domains is None:
            return X

        ids = {d: i for i, d in enumerate(sorted(set(domains)))}
        rows = np.fromiter((ids[d] for d in domains), dtype=np.int64,
                           count=len(domains))
        # group rows by domain, then count how many domains hit each feature
        group = csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))),
                           shape=(len(ids), X.shape[0]))
        per_domain = (group @ (X > 0)).tocsc()
        domain_df = np.diff(per_domain.indptr)

        self.keep_ = domain_df >= self.min_domain_df
        return X[:, self.keep_]

    def transform(self, docs):
        X = self.vec.transform(docs)
        return X if self.keep_ is None else X[:, self.keep_]

    def get_feature_names_out(self):
        names = self.vec.get_feature_names_out()
        return names if self.keep_ is None else names[self.keep_]


def url_features():
    """Character 3-5 grams over the URL string. The shipped model."""
    return DomainPurityTfidf(analyzer="char", ngram_range=(3, 5), min_df=2)


def text_features(min_domain_df=MIN_DOMAIN_DF):
    """Words and word pairs over the extracted page body.

    Bigrams earn their cost here because legal register is collocational:
    'court' is on every court website, 'the court held' is an opinion.
    Trigrams are left out - at ~4k documents they are mostly singletons that
    min_df prunes anyway. No stop-word list, because dropping 'to' turns
    'pursuant to' into 'pursuant' and throws away the signal bigrams exist
    to catch; idf already handles genuinely uninformative words.
    """
    return DomainPurityTfidf(min_domain_df=min_domain_df, analyzer="word",
                             ngram_range=(1, 2), min_df=3,
                             max_features=200_000, sublinear_tf=True)


# mode -> (vectorizer factory, source file, field holding the text to vectorize)
MODES = {
    "url": (url_features, LABELED_URLS, "url"),
    "text": (text_features, LABELED_TEXT, "text"),
}


def make_classifier():
    """The model itself. class_weight balanced because legal is about a
    quarter of the labels and a fifth of the domains; without it the fit
    optimises the majority class and the threshold sweep has nothing to
    trade off."""
    return LogisticRegression(class_weight="balanced", max_iter=2000)


def domain_weights(labels, domains, alpha):
    """Per-row weights that stop a few large publishers from dominating a fit.

    Three publishers hold most of the legal rows, so most of what a fit learns
    about legal text comes from those three. Each row gets a weight based on
    how many rows its publisher contributed to its own class: a publisher with
    342 legal rows gets a small weight on each of them, a publisher with 5
    gets a large one.

    alpha picks how far to go. 0 leaves every weight at 1, which is an
    unweighted fit. 1 makes every publisher's rows sum to the same total
    within its class, so each publisher counts once no matter its size.
    Values in between are partial, and 0.5 works out to the square root of
    the row count.

    Weights are rescaled to average 1. Without that the total weight changes
    with alpha, which changes how hard the regularisation bites, and two runs
    would differ for two reasons at once instead of one.

    Returns None at alpha 0 so callers can pass the result straight to
    sklearn's sample_weight, where None means unweighted.
    """
    if not alpha:
        return None
    counts = Counter(zip(domains, labels))
    w = np.array([(1.0 / counts[(d, y)]) ** alpha
                  for d, y in zip(domains, labels)], dtype=float)
    return w * (len(w) / w.sum())


def domain_of(url):
    """Registered domain, so law.cornell.edu and www.law.cornell.edu group
    together as one publisher."""
    return _extract(urlparse(url).netloc or url).top_domain_under_public_suffix


def load_bundle(path, quiet=False):
    """Load a model saved by train_classifier.py.

    Scripts that need a model load one from here instead of fitting their own,
    so a reported number always names the model that produced it. Fitting
    inline is how four scripts ended up carrying four copies of the recipe.
    """
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} - run src/classifier/train_classifier.py first")
    bundle = joblib.load(path)
    if not quiet:
        meta = bundle["meta"]
        print(f"--- model: {path} ---")
        print(f"mode {bundle['mode']}, threshold {bundle['threshold']}, "
              f"{meta['n_features']} features")
        print(f"trained {meta['trained_at']} on {meta['n_train']} rows "
              f"from {meta['label_file']}")
        if meta.get("extractors"):
            print(f"training text extracted with: {', '.join(meta['extractors'])}")
    return bundle


def legal_probs(bundle, docs):
    """Probability of the legal class for each document."""
    clf = bundle["classifier"]
    idx = list(clf.classes_).index("legal")
    return clf.predict_proba(bundle["vectorizer"].transform(docs))[:, idx]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_labeled(mode, path=None, quiet=False):
    """Load one mode's labeled rows as parallel lists.

    Returns (urls, docs, labels, domains, path). urls is kept alongside docs
    because in text mode the two differ: the model reads the page body, but
    grouping and auditing still need to know which publisher a row came from.

    Rows whose extraction failed carry a null text and are dropped here, so
    they can't vectorize as empty strings and count as documents.
    """
    _, default_path, field = MODES[mode]
    path = path or default_path
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} - run src/common/fetch_warc_text.py first")

    by_url, dropped = {}, 0
    for obj in read_jsonl(path):
        doc = obj.get(field)
        if not doc:
            dropped += 1
            continue
        by_url.setdefault(obj["url"], (doc, obj["label"]))

    urls = list(by_url)
    docs = [by_url[u][0] for u in urls]
    labels = [by_url[u][1] for u in urls]
    domains = [domain_of(u) for u in urls]
    if dropped and not quiet:
        print(f"  dropped {dropped} rows with no {field}")
    return urls, docs, labels, domains, path

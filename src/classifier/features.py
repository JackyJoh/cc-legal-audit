"""
The one thing that differs between the URL classifier and the content
classifier: what the model is allowed to look at.

Both models share everything else - the same labels, the same splits, the
same leave-one-domain-out loop, the same metrics. Keeping the two recipes
here means train and eval can't drift onto different settings and quietly
stop comparing like with like, which is the whole point of running them
against each other.

"url" is the existing char n-gram recipe, unchanged, so the numbers it
produces stay comparable to everything already reported. "text" reads the
extracted page body from labeled_text.jsonl instead.
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

LABELED_URLS = "data/processed/labeled_urls.jsonl"
LABELED_TEXT = "data/processed/labeled_text.jsonl"

# Drop any n-gram confined to fewer than this many registered domains. Set
# from the observed gap in domain-DF rather than tuned against a score:
# Justice Canada's page template sits at 1-2 domains ('marginal note' 1,
# 'details date' 1, 'date modified' 2 across justice.gc.ca and crtc.gc.ca),
# while genuine register vocabulary starts at 17 ('general assembly') and
# runs to 79 ('section'). 3 is the smallest value that clears all four
# template phrases; 4 drops vocabulary retention off a cliff, 72% -> 49%.
MIN_DOMAIN_DF = 3


class DomainPurityTfidf:
    """TF-IDF that discards features living on too few distinct publishers.

    A phrase appearing on many pages of one site is that site's template; a
    phrase appearing across many sites is register. Counting *publishers*
    rather than documents is the difference, and it needs no site list, so a
    new publisher's boilerplate is handled without anyone naming it.

    domains must be the training fold's domains and nothing else - counting
    over the full dataset would let a held-out publisher decide which
    features exist, which is the leak the grouped and LODO splits exist to
    detect.
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

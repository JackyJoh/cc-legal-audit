"""
Reports how long legal documents are against non-legal pages, over every
hand-labeled page that has extracted text. Answers one question before corpus
construction: is the length gap between the two buckets large enough that a
13-gram MinHash sweep would be comparing documents of a different shape, not
just a different register.

Two labeled sets exist and they are different populations, so both are
reported separately and then pooled:

1. The training label set (labeled_text.jsonl). Five sampling passes mixed
   together: open-web random, whitelist hits, publisher hosts, bill sections
   and model-flagged pages. Its non_legal side is dominated by open-web pages,
   and 60% of its legal side is cornell.edu and justice.gc.ca, neither of
   which is on the authority list. Unweighted.

2. The precision batch (legal_sample_batch.jsonl). 250 pages from inside the
   authority domains, drawn as three score strata at different rates, so its
   raw distribution over-represents high-score pages. Reported raw and then
   reweighted by stratum, each label standing for frame / labels-drawn pool
   pages, using the same 'available' frame precision_report.py uses. The
   reweighted read is the deployment estimate.

non_legal is split two more ways. First, pages whose host is on
legal_domains.jsonl (dockets, bill status pages, indexes) against pages that
are not. Membership uses each row's own sampling unit, exactly as
fetch_legal_pool.py samples: a court row matches its hosts, a legislature row
matches its registered domain, both under the bare and www. spellings.
Second, and narrower, the raw_random slice of the original batch: a uniform
draw of crawl pages that the rule-based prefilter did not flag, which at its
0.2% hit rate removes almost nothing. Off-domain still carries whitelist
rejects and targeted pulls, so raw_random is the only slice that stands in
for the general web as a uniform page sample.

Length is whitespace-token count on the trafilatura output, which is also the
unit a word-shingle MinHash sees. Text was capped at 500,000 chars by
fetch_warc_text.py, so the extreme tail is understated; medians are
unaffected.

Reads:
  data/processed/labeled_text.jsonl              training labels with text
  data/candidates/legal_sample_batch.jsonl       precision batch, hand labels
  data/candidates/legal_sample_scores.jsonl      score and stratum per URL
  data/candidates/legal_sample_stats.json        per-stratum frame sizes
  data/candidates/_legal_pool_text_cache.jsonl   text for the precision batch
  data/candidates/legal_domains.jsonl            authority list, for the split
  data/candidates/candidates.jsonl               original batch, for its hints

Writes nothing; the tables are the output.
"""
import json
from collections import Counter
from urllib.parse import urlparse

import tldextract

TRAIN_FILE   = "data/processed/labeled_text.jsonl"
BATCH_FILE   = "data/candidates/legal_sample_batch.jsonl"
SCORES_FILE  = "data/candidates/legal_sample_scores.jsonl"
STATS_FILE   = "data/candidates/legal_sample_stats.json"
POOL_TEXT    = "data/candidates/_legal_pool_text_cache.jsonl"
DOMAINS_FILE = "data/candidates/legal_domains.jsonl"
CANDS_FILE   = "data/candidates/candidates.jsonl"

QUANTILES = ((0.10, "p10"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.90, "p90"))
SHORT_CUT = 200                                    # the README's short-document line
HIST_EDGES = (0, 100, 200, 400, 800, 1600, 3200, 6400)   # log2 word bins
BAR_WIDTH = 24                                     # chars for a 100% bar

# pinned local suffix list, no PSL fetch at runtime
_extract = tldextract.TLDExtract(suffix_list_urls=())


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def registered_domain(host):
    return _extract(host).top_domain_under_public_suffix


def spellings(host):
    """Both spellings CC might have stored this host under."""
    bare = host[4:] if host.startswith("www.") else host
    return {bare, "www." + bare}


def build_membership(domains):
    """Host set for court rows, registered-domain set for legislature rows."""
    hosts, regs = set(), set()
    for dom in domains:
        for host in dom["hosts"]:
            hosts |= spellings(host)
        if dom["unit"] == "registered_domain":
            regs.add(dom["registered_domain"])
    return hosts, regs


def on_legal_domain(url, hosts, regs):
    host = urlparse(url).hostname or ""
    return host in hosts or registered_domain(host) in regs


def words(row):
    return len(row["text"].split())


def weighted_quantile(values, weights, q):
    """Smallest value at or past the q-th share of total weight."""
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= q * total:
            return value
    return pairs[-1][0]


def summary_row(rows, weights=None):
    """Quantiles, mean and short-doc share of word count, weighted if asked."""
    w = weights if weights is not None else [1.0] * len(rows)
    total = sum(w)
    n_words = [words(r) for r in rows]
    return {
        "n": len(rows),
        **{name: weighted_quantile(n_words, w, q) for q, name in QUANTILES},
        "mean": sum(v * x for v, x in zip(n_words, w)) / total,
        "short": 100 * sum(x for v, x in zip(n_words, w) if v < SHORT_CUT) / total,
    }


def print_summary(title, groups):
    """One table: a row per group, columns are the word-count quantiles."""
    cols = [name for _, name in QUANTILES]
    print(f"\n{title}")
    print(f"  {'group':<32}{'n':>6}" + "".join(f"{c:>8}" for c in cols)
          + f"{'mean':>8}{'<' + str(SHORT_CUT) + 'w':>7}")
    print("  " + "-" * (32 + 6 + 8 * len(cols) + 8 + 7))
    for label, rows, weights in groups:
        if not rows:
            print(f"  {label:<32}{'-':>6}")
            continue
        s = summary_row(rows, weights)
        print(f"  {label:<32}{s['n']:>6,}" + "".join(f"{s[c]:>8,}" for c in cols)
              + f"{s['mean']:>8,.0f}{s['short']:>6.0f}%")


def print_histogram(title, groups):
    """Share of each group per log-scale word bin, drawn as bars."""
    edges = list(HIST_EDGES) + [float("inf")]
    print(f"\n{title}")
    print(f"  {'words':>10}  " + "".join(f"{label:<{BAR_WIDTH + 8}}" for label, _ in groups))
    for lo, hi in zip(edges, edges[1:]):
        bin_label = f"{lo}-{hi - 1}" if hi != float("inf") else f"{lo}+"
        cells = []
        for _, rows in groups:
            share = 100 * sum(1 for r in rows if lo <= words(r) < hi) / len(rows)
            bar = "#" * round(share / 100 * BAR_WIDTH)
            cells.append(f"{share:>4.0f}%  {bar:<{BAR_WIDTH}}")
        print(f"  {bin_label:>10}  " + "".join(f"{c:<{BAR_WIDTH + 8}}" for c in cells))


def main():
    hosts, regs = build_membership(load_jsonl(DOMAINS_FILE))

    def is_legal_domain(row):
        return on_legal_domain(row["url"], hosts, regs)

    # ---- training label set ----
    train = [r for r in load_jsonl(TRAIN_FILE) if r.get("text")]
    t_legal = [r for r in train if r["label"] == "legal"]
    t_non = [r for r in train if r["label"] == "non_legal"]
    t_non_off = [r for r in t_non if not is_legal_domain(r)]
    t_non_on = [r for r in t_non if is_legal_domain(r)]

    # the original batch's hint says which of its three buckets a URL came
    # from; raw_random is the uniform crawl draw
    hint_of = {r["url"]: r["hint"] for r in load_jsonl(CANDS_FILE)}
    t_non_random = [r for r in t_non
                    if r["source"] == "original" and hint_of.get(r["url"]) == "raw_random"]

    # ---- precision batch ----
    pool_text = {r["url"]: r["text"] for r in load_jsonl(POOL_TEXT) if r.get("text")}
    stratum_of = {r["url"]: r["stratum"] for r in load_jsonl(SCORES_FILE)}
    with open(STATS_FILE, encoding="utf-8") as f:
        strata = json.load(f)["strata"]

    batch = []
    for b in load_jsonl(BATCH_FILE):
        if b["label"] in ("legal", "non_legal") and b["url"] in pool_text:
            batch.append({"url": b["url"], "label": b["label"],
                          "text": pool_text[b["url"]], "stratum": stratum_of[b["url"]]})

    # weight = frame / labels drawn, per stratum; frame is 'available', the
    # pool minus pages already labeled in earlier batches, as precision_report
    drawn = Counter(r["stratum"] for r in batch)
    frame = {name: st["available"] for name, st in strata.items()}

    def weight(row):
        return frame[row["stratum"]] / drawn[row["stratum"]]

    p_legal = [r for r in batch if r["label"] == "legal"]
    p_non = [r for r in batch if r["label"] == "non_legal"]
    p_legal_w = [weight(r) for r in p_legal]
    p_non_w = [weight(r) for r in p_non]

    # ---- pooled ----
    a_legal = t_legal + p_legal
    a_non = t_non + p_non
    a_non_off = [r for r in a_non if not is_legal_domain(r)]

    print("DOCUMENT LENGTH, words per page")

    print_summary("Training label set (five passes pooled, unweighted)", [
        ("legal",                       t_legal,   None),
        ("non_legal, all",              t_non,     None),
        ("non_legal, off legal_domains", t_non_off, None),
        ("non_legal, on legal_domains",  t_non_on,  None),
        ("non_legal, raw_random only",   t_non_random, None),
    ])
    print_summary("Precision batch (inside legal domains; weighted = deployment estimate)", [
        ("legal, raw",                  p_legal, None),
        ("non_legal, raw",              p_non,   None),
        ("legal, weighted",             p_legal, p_legal_w),
        ("non_legal, weighted",         p_non,   p_non_w),
    ])
    print_summary("Pooled (both sets, unweighted)", [
        ("legal",                       a_legal,   None),
        ("non_legal, all",              a_non,     None),
        ("non_legal, off legal_domains", a_non_off, None),
        ("non_legal, raw_random only",   t_non_random, None),
    ])

    print_histogram("Pooled, share of docs per word bin", [
        ("legal", a_legal),
        ("non_legal, all", a_non),
        ("non_legal, raw_random only", t_non_random),
    ])

    # ---- provenance notes a reader needs to weigh the tables ----
    legal_off = [r for r in t_legal if not is_legal_domain(r)]
    capped = sum(1 for r in train if len(r["text"]) >= 500_000)
    print("\nNotes")
    print(f"  training legal rows off legal_domains : {len(legal_off)} / {len(t_legal)}  "
          + ", ".join(f"{h} {n}" for h, n in
                      Counter(urlparse(r["url"]).hostname for r in legal_off).most_common(3)))
    print(f"  training non_legal on legal_domains   : {len(t_non_on)}, top hosts "
          + ", ".join(f"{h} {n}" for h, n in
                      Counter(urlparse(r["url"]).hostname for r in t_non_on).most_common(3)))
    print(f"  training non_legal by pass            : "
          + ", ".join(f"{p} {n}" for p, n in Counter(r["source"] for r in t_non).most_common()))
    print(f"  original non_legal by hint            : "
          + ", ".join(f"{h} {n}" for h, n in Counter(
              hint_of.get(r["url"]) for r in t_non if r["source"] == "original").most_common()))
    print(f"  precision strata (frame / drawn)      : "
          + ", ".join(f"{s} {frame[s]}/{drawn[s]}" for s in ("high", "mid", "low")))
    print(f"  precision non_legal off legal_domains : 0 by construction")
    print(f"  URL overlap between the two sets      : "
          f"{len({r['url'] for r in train} & {r['url'] for r in batch})}")
    print(f"  rows at the 500,000-char text cap     : {capped}")


if __name__ == "__main__":
    main()

# cc-legal-audit

Empirical audit of how uniform MinHash fuzzy deduplication thresholds affects semantic coverage (topic entropy) in the legal domain versus general web text, using Common Crawl data.

## Motivation

Standard LLM pre-training pipelines apply a uniform Jaccard similarity threshold (typically 0.8 on 13-grams) inherited from Gopher without empirical validation across domains. Legal text has constrained vocabulary and structural conventions that inflate n-gram similarity scores. Substantively different documents get flagged as duplicates not because they share content, but because they sound alike. DCLM noted this as an open problem and left domain-level investigation as future work. This project picks that up.

## Methodology

1. Sample a Common Crawl snapshot (CC-MAIN-2026-12)
2. Apply fixed preprocessing: language filtering, quality heuristics, repetition removal (Gopher defaults held constant)
3. Split pages into a legal bucket (see [Legal detection](#legal-detection)) and a general-web bucket
4. Measure baseline topic entropy (BERTopic + Shannon entropy) per domain before deduplication
5. Run MinHash fuzzy dedup at Jaccard thresholds 0.6, 0.7, 0.8, 0.9
6. Re-measure topic entropy per domain after each threshold
7. Compare coverage loss curves across domains to quantify asymmetry

## Legal detection

A page counts as legal only if the page itself carries the text of a statute, bill, regulation, filing, or court opinion, published by a primary source (court, legislature, agency, or established legal publisher). Commentary, news, law-firm pages, and index pages that only link to the text don't count.

Detection is a two-stage cascade:

- **Jev decides.** An LLM classifier ([TypeSafe](https://typesafe.ai) Jev), asked one yes/no question with the full definition above, one page per request. A page is legal if Jev scores it at **J = 0.90** or higher. On a uniform open-web draw, all 160 pages above that cut were hand-verified legal (precision 95% CI [0.977, 1.0]); precision falls to about 0.89 by 0.60.
- **TF-IDF screens.** The older TF-IDF + logistic regression model runs first, only to cut the number of Jev calls. Its precision doesn't matter, only its recall: a legal page it rejects never reaches Jev. T starts at **0.37**, the lowest TF-IDF score among the Jev-accepted legal pages, so nothing measured is lost there.

**Choosing T.** Quality filters and TF-IDF run over the whole sample first (both local, free). The Jev cost at each T is then known before any Jev call, and T is set to the lowest value that fits the budget (~$50). Once T is used on the corpus it is frozen.

| T | legal lost (of 160) | Jev cost, 100k legal docs |
|---|---|---|
| **0.37** | 0 | $66–85 |
| 0.40 | 1 | $54–69 |
| 0.45 | 3 | $38–49 |
| 0.50 | 5 | $29–38 |

Cost ranges run from the measured token rate to a pessimistic bound; real cost should land lower, since quality filters shrink both page count and text before Jev sees it. About 0.16% of crawl pages come out legal.

**Caveat.** TF-IDF misses short documents first (single-section statutes and regulations), so every step up from 0.37 skews the legal corpus slightly toward longer documents. No document type drops out anywhere on this ladder.

Full design history, the dropped rule-based and URL-only approaches, the original TF-IDF evaluation, and every intermediate number: [`src/classifier/README.md`](src/classifier/README.md).

## Code

```
src/
  common/      shared Common Crawl access: Athena querying, WARC fetch/extract
  classifier/  legal-document classifier: sourcing, labels, model, validation
```

Details in [`src/classifier/README.md`](src/classifier/README.md).

## Paper
University of Florida undergraduate research.

# cc-legal-audit

Empirical audit of how uniform MinHash fuzzy deduplication thresholds affects semantic coverage (topic entropy) in the legal domain versus general web text, using Common Crawl data.

## Motivation

Standard LLM pre-training pipelines apply a uniform Jaccard similarity threshold (typically 0.8 on 13-grams) inherited from Gopher without empirical validation across domains. Legal text has constrained vocabulary and structural conventions that inflate n-gram similarity scores. Substantively different documents get flagged as duplicates not because they share content, but because they sound alike. DCLM noted this as an open problem and left domain-level investigation as future work. This project picks that up.

## Methodology

1. Sample a Common Crawl snapshot (CC-MAIN-2026-12)
2. Apply fixed preprocessing: language filtering, quality heuristics, repetition removal (Gopher defaults held constant)
3. Source the legal bucket from authority-enumerated legal publishers, then classify page type within it on page text; draw the general bucket from depth-matched random hosts
4. Measure baseline topic entropy (BERTopic + Shannon entropy) per domain before deduplication
5. Run MinHash fuzzy dedup at Jaccard thresholds 0.6, 0.7, 0.8, 0.9
6. Re-measure topic entropy per domain after each threshold
7. Compare coverage loss curves across domains to quantify asymmetry

## Classifier

The classifier's job: given a page, decide if it's an actual legal document (a statute, bill, regulation, or court opinion with the text on the page itself), not a page that just links to one. It's a TF-IDF (word 1-2 grams) + logistic regression model over the page's extracted text, trained on ~4,600 hand-labeled pages across 42 legal-bearing domains.

The classifier only scores pages that already sit on an authority-enumerated list of legal publishers (courts, legislatures), not random pages pulled from the open web. On those domains, about 1 in 3 pages is legal, so its hits are reliable. Scored against the whole crawl instead, legal pages are only about 1 in 800 (measured at 0.12%), so even a fairly accurate model returns mostly wrong hits: precision stayed below 0.7 at any usable recall.

**Performance**, measured on a stratified hand-labeled sample of 250 pages drawn from inside the legal domains (2026-09-08):

| threshold | precision | contamination | recall |
|---|---|---|---|
| 0.60 | 0.911 | 8.9% | 0.770 |
| **0.75 (operating)** | **0.953** | **4.7%** | **0.602** |
| 0.85 | 0.980 | 2.0% | 0.418 |

Contamination is the share of kept pages that are actually non-legal. At crawl scale, the 0.75 threshold keeps ~140k pages, ~134k of them legal.

Closed as of 2026-09-08. Full design history, the dropped rule-based and URL-only approaches, and every intermediate number: [`src/classifier/README.md`](src/classifier/README.md).

## Code

```
src/
  common/      shared Common Crawl access: Athena querying, WARC fetch/extract
  classifier/  legal-document classifier: sourcing, labels, model, validation
```

Details in [`src/classifier/README.md`](src/classifier/README.md).

## Paper
University of Florida undergraduate research.

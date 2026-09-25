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

Detection is a two-stage cascade, both stages local and free:

- **Laya decides.** A 421M open-weights decision model ([Laya](https://huggingface.co/convaiinnovations/laya)), fine-tuned to reproduce a commercial LLM classifier ([TypeSafe](https://typesafe.ai) Jev) on ~30k pages Jev scored against the definition above. A page is legal if Laya scores it at **0.85** or higher. On the ~76k open-web pages it never trained on, all 172 pages above that cut were hand-verified legal (precision 95% CI [0.978, 1.0]), matching Jev itself (160/160 at its 0.90 cut). Model and cut are frozen.
- **TF-IDF screens.** The older TF-IDF + logistic regression model runs first, only to cut how many pages Laya has to score (~72 pages/s on one consumer GPU). Its precision doesn't matter, only its recall: a legal page it rejects never reaches Laya. T is the highest value that loses no known legal page (the lowest measured so far is 0.37), set once over the filtered sample and then frozen.

About 0.16% of crawl pages come out legal.

**Caveats.** Both stages lose short documents first (single-section statutes and regulations), which skews the legal corpus slightly toward longer documents. Laya is also weakest on decisions: of the legal pages Jev keeps, it misses some tribunal decisions (3 of 6 WIPO domain decisions in the test).

Full design history, how Jev and Laya were tested, the dropped rule-based and URL-only approaches, the original TF-IDF evaluation, and every intermediate number: [`src/classifier/README.md`](src/classifier/README.md).

## Code

```
src/
  common/      shared Common Crawl access: Athena querying, WARC fetch/extract
  classifier/  legal-document classifier: sourcing, labels, model, validation
```

Details in [`src/classifier/README.md`](src/classifier/README.md).

## Paper
University of Florida undergraduate research.

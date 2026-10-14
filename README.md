# cc-legal-audit

Empirical audit of how uniform MinHash fuzzy deduplication thresholds affects semantic coverage (topic entropy) in the legal domain versus general web text, using Common Crawl data.

## Motivation

Standard LLM pre-training pipelines apply a uniform Jaccard similarity threshold (typically 0.8 on 13-grams) inherited from Gopher without empirical validation across domains. Legal text has constrained vocabulary and structural conventions that inflate n-gram similarity scores. Substantively different documents get flagged as duplicates not because they share content, but because they sound alike. DCLM noted this as an open problem and left domain-level investigation as future work. This project picks that up.

## Methodology

1. Sample a Common Crawl snapshot (CC-MAIN-2026-12): 26,776 WARC files, evenly across all 100 segments, 99.4M pages after filtering
2. Apply fixed preprocessing: English only, a URL blocklist, Gopher quality and repetition rules, C4's lorem-ipsum and curly-bracket rules, then exact copies (same URL or same text) removed. Exact copies count as filtering, not deduplication
3. Split pages into a legal bucket (see [Legal detection](#legal-detection)) and a general-web bucket
4. Fit topics once per bucket (bge-small-en-v1.5 embeddings of the first 512 tokens, BERTopic) and freeze every page's topic label
5. Run MinHash fuzzy dedup at Jaccard thresholds 0.6, 0.7, 0.8, 0.9: 128 hashes on word 13-grams, LSH with 32 bands of 4, every candidate pair checked by exact Jaccard, near-duplicate groups by connected components
6. After each threshold, recount topics with the same frozen labels (no refit, which could hide lost topics), and compare against removing the same number of pages at random
7. The headline is the gap: legal's loss beyond random minus general's loss beyond random, in effective topics (2^entropy), at three topic granularities, with 95% bootstrap intervals over WARC files

## Legal detection

A page counts as legal only if the page itself carries the text of a statute, bill, regulation, filing, or court opinion, published by a primary source (court, legislature, agency, or established legal publisher). Commentary, news, law-firm pages, and index pages that only link to the text don't count.

Detection is a two-stage cascade, both stages local and free:

- **Laya decides.** A 421M open-weights decision model ([Laya](https://huggingface.co/convaiinnovations/laya)), fine-tuned to reproduce a commercial LLM classifier ([TypeSafe](https://typesafe.ai) Jev) on ~30k pages Jev scored against the definition above. A page is legal if Laya scores it at **0.85** or higher. On the ~76k open-web pages it never trained on, all 172 pages above that cut were hand-verified legal (precision 95% CI [0.978, 1.0]), matching Jev itself (160/160 at its 0.90 cut). Model and cut are frozen.
- **TF-IDF screens.** The older TF-IDF + logistic regression model runs first, only to cut how many pages Laya has to score (~72 pages/s on one consumer GPU). Its precision doesn't matter, only its recall: a legal page it rejects never reaches Laya. **T = 0.30**, frozen.

About 0.05% of pages that pass the quality filters come out legal: 50,104 of 86.3M in the full run, after exact copies. (The classifier work measured ~0.16% of English HTML pages before the filters.)

Full design history, how Jev and Laya were tested, the dropped rule-based and URL-only approaches, the original TF-IDF evaluation, and every intermediate number: [`src/classifier/README.md`](src/classifier/README.md).

## MinHash pilot

Before the full run, a hand-inspected check of whether MinHash's legal near-duplicates are real copies or different documents with shared structure. Among pairs a 0.7 threshold would remove, 50% of legal pairs were distinct content against 19% of a size-matched control (+31 points): preliminary evidence of asymmetric deduplication. In this sample, most of that gap comes from one statute publisher, whose section pages show each section inside the surrounding chapter text, so neighbouring sections overlap and look like duplicates. That is the structure-driven confusion the pilot was built to detect, occurring on a real legal publisher; how common it is across legal publishers is more than a 110-pair sample can show. Result: go. Method, results and caveats: [`src/corpus/minhash_pilot/README.md`](src/corpus/minhash_pilot/README.md).

## Full run

26,776 WARC files from CC-MAIN-2026-12, spread evenly across all 100 segments: 99,415,529 pages after filtering, 86,301,929 after exact copies (86,251,825 general, 50,104 legal).

Near-duplicates removed:

| Threshold | General | Legal |
|---|---|---|
| 0.6 | 11.4% | 6.5% |
| 0.7 | 8.8% | 5.2% |
| 0.8 | 6.0% | 3.9% |
| 0.9 | 3.0% | 2.2% |

Topics: 218 general and 283 legal at the finest level.

Built in ~55 h on one HiPerGator job of 32 CPUs + 1 RTX 6000. Per-step timings and how they were measured: [`src/corpus/README.md`](src/corpus/README.md).

## Code

```
src/
  common/      shared Common Crawl access: Athena querying, WARC fetch/extract
  classifier/  legal-document classifier: sourcing, labels, model, validation
  corpus/      the full run: sourcing + filters, labeling, MinHash dedup, topics + entropy, the MinHash pilot
jobs/          HiPerGator batch scripts for the full run
```

Details in [`src/classifier/README.md`](src/classifier/README.md) and, for the full run and its timings, [`src/corpus/README.md`](src/corpus/README.md).

## Paper
University of Florida undergraduate research.

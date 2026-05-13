# cc-legal-audit

Empirical audit of how uniform MinHash fuzzy deduplication thresholds affects semantic coverage (topic entropy) in the legal domain versus general web text, using Common Crawl data.

## Motivation

Standard LLM pre-training pipelines apply a uniform Jaccard similarity threshold (typically 0.8 on 13-grams) inherited from Gopher without empirical validation across domains. Legal text has constrained vocabulary and structural conventions that inflate n-gram similarity scores — substantively different documents get flagged as duplicates not because they share content, but because they sound alike. DCLM noted this as an open problem and left domain-level investigation as future work. This project picks that up.

## Methodology

1. Sample a Common Crawl snapshot (CC-MAIN-2026-17)
2. Apply fixed preprocessing: language filtering, quality heuristics, repetition removal (Gopher defaults held constant)
3. Classify documents into legal and general web subsets via URL tokenization
4. Measure baseline topic entropy (BERTopic + Shannon entropy) per domain before deduplication
5. Run MinHash fuzzy dedup at Jaccard thresholds 0.6, 0.7, 0.8, 0.9
6. Re-measure topic entropy per domain after each threshold
7. Compare coverage loss curves across domains to quantify asymmetry

## Code

- `url_classifier.py` — Classifies any URL as legal or non-legal via keyword token matching
- `recall_test.py` — Validates the classifier against labeled legal documents

## Paper
University of Florida undergraduate research.

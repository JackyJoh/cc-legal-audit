# cc-legal-audit

Empirical audit of how uniform MinHash fuzzy deduplication thresholds affects semantic coverage (topic entropy) in the legal domain versus general web text, using Common Crawl data.

## Motivation

Standard LLM pre-training pipelines apply a uniform Jaccard similarity threshold (typically 0.8 on 13-grams) inherited from Gopher without empirical validation across domains. Legal text has constrained vocabulary and structural conventions that inflate n-gram similarity scores. Substantively different documents get flagged as duplicates not because they share content, but because they sound alike. DCLM noted this as an open problem and left domain-level investigation as future work. This project picks that up.

## Methodology

1. Sample a Common Crawl snapshot (CC-MAIN-2026-17)
2. Apply fixed preprocessing: language filtering, quality heuristics, repetition removal (Gopher defaults held constant)
3. Classify documents into legal and general web subsets via URL tokenization
4. Measure baseline topic entropy (BERTopic + Shannon entropy) per domain before deduplication
5. Run MinHash fuzzy dedup at Jaccard thresholds 0.6, 0.7, 0.8, 0.9
6. Re-measure topic entropy per domain after each threshold
7. Compare coverage loss curves across domains to quantify asymmetry

### URL Classifier

1. Register the CC columnar index (`ccindex`) as an Athena table to enable SQL queries over a given crawl snapshot. 
2. Run `WL_Builder.py` against an OOS snapshot (CC-MAIN-2026-12, separate from the research snapshot) to discover candidate legal domains. The query groups by `url_host_name`, filters to English pages, and retains only hosts with ≥1000 pages to exclude low-traffic noise.
3. Manually triage the candidate list: keep a domain only if its primary function is producing or publishing formal legal documents (court records, statutes, regulations). Advocacy organizations, legal news, and commentary are excluded.
4. Add a keyword fallback for domains absent from the whitelist. The hostname is tokenized (path ignored) and each token matched exactly against a conservative list of legal terms. Light stemming handles plurals. Keyword fallback is restricted to non-commercial TLDs (`.gov`, `.ca`, etc.) — `.com`, `.edu`, `.org`, `.net` are excluded. Keywords pruned iteratively on the OOS snapshot until FP rate is acceptable.
5. Measure precision and recall on the research snapshot (CC-MAIN-2026-17) with seed 42: manually label ~200–300 positives from a 100k URL sample, and scan a fixed 75-URL negative sample for false negatives.
6. External validation by performing classification against labeled legal urls from court-listener bulk data. First (un-touched WL) run yielded 84.2% recall; adding highest missed domains and running again on new random sample yielded 88.4% recall. Note: not all domains from first run were added to WL due to irregularity and low density of legal documents (i.e., mostly information & directories instead of legal documentation). 

## Code

- `URL_Classifier.py`: URL-based legal/non-legal classifier. Two-layer architecture: curated domain whitelist (`wl_candidates.txt`) checked first, then strict keyword matching on the hostname only (path ignored to prevent false positives).
- `WL_Builder.py`: Discovers candidate legal domains from a CC snapshot via Athena. Queries `url_host_name` grouped by page count and writes results to `wl_candidates.txt` for manual triage.
- `wl_candidates.txt`: Triaged whitelist of primary legal source domains (courts, legislatures, statute repositories). One entry per line; suffix matching at runtime covers all subdomains.
- `CC_Classifier_Test.py`: Samples URLs from a CC snapshot via Athena TABLESAMPLE, classifies them, and prints positives tagged `[WL]` or `[KW]` plus a negative sample for manual precision/recall review.

## Paper
University of Florida undergraduate research.

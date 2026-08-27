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

#### Rule-based approach (archived)

Curated domain whitelist + a conservative hostname-keyword fallback (details in `archive/rule-based/`), validated at 88.4% recall against CourtListener bulk data after one round of gap-filling.

**Why this was dropped.** Both layers are hardcoded human judgments about which domains/terms "count," not something learned from data. Closing a recall gap meant manually finding and adding a domain, a process that doesn't scale and has to be redone per snapshot or jurisdiction. More fundamentally, hostname-level matching can't distinguish page *type*: it treats a bare homepage or a case-listing index page the same as an actual statute or opinion page on the same domain, since that distinction lives in the URL path and content, not the hostname. And a binary keep/reject decision gives no confidence signal to trade precision against recall. This motivated the pivot to a model trained on the actual distribution of legal vs. non-legal URLs.

#### Current approach: char n-gram TF-IDF + logistic regression

1. **Ground-truth labeling pipeline.** Wrote a labeling task prompt (`prompts/legal_url_labeling_task.md`) that has a labeling agent fetch and read each candidate page and judge it against a strict LEGAL/NON_LEGAL definition, grounded in the page's own content, not URL heuristics. LEGAL requires the source's primary function to be producing/publishing formal legal documents *and* the specific page's own HTML to contain the actual text of a filing, statute, bill, regulation, or court opinion; a page that merely links out to that text (a landing/index page) is NON_LEGAL, since Common Crawl only captures a page's own HTML, never what it links to.
2. **Candidate sourcing** (`fetch_candidate_urls.py` + `build_label_batch.py`): pulled a 600k-URL raw pool via Athena, then built a 2990-URL candidate batch mixing rule-based prefilter hits (to boost legal density, since legal pages are ~0.2–0.3% of the raw crawl), raw random URLs, and synthetic homepage URLs for whitelisted domains (deliberate hard negatives for the index-vs-filing failure mode).
3. **Labeling run**: 6 parallel instances of the labeling agent worked the full candidate batch, producing 2774 labeled URLs (756 legal / 2018 non_legal) plus 216 skipped (source blocked automated fetches with no accessible mirror). This is a single automated labeling policy (same model/prompt) applied at scale with human audit after the fact, not independent human annotators.
4. **Audit and correction** (`intake.py`): manual review of the legal-labeled set caught two systematic errors under the original (looser) definition (18 link-only pages and 16 index/landing pages templated across a URL family by one worker), corrected to non_legal, which is what drove tightening the labeling definition in step 1. A follow-up grep-by-rationale-keyword audit of all 2774 labels found no further drift.
5. **Training** (`train_classifier.py`): char n-gram (3–5) TF-IDF vectorizer + logistic regression, fit on the corrected label set with an 80/20 stratified held-out split (seed 42). URL string only as the feature: no page content, no hardcoded domain or keyword lists.
6. **Threshold selection**: chose an operating threshold of 0.9 on predicted legal-probability, prioritizing precision over recall. False positives pollute the small legal bucket the downstream topic-diversity analysis depends on; false negatives are reabsorbed into the much larger non-legal bucket where they're a negligible rounding error (raising the threshold from 0.6 to 0.9 drops only 0.067% of the non-legal pool while cutting false positives by more than half).

**Results** (held-out test set, 654 URLs: 207 legal / 447 non_legal, after merging in the error-driven +500 batch; full sweep in `data/processed/threshold_sweep_results.csv`):

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.5 | 0.768 | 0.961 | 0.854 |
| 0.65 (F1-optimal) | 0.823 | 0.918 | 0.868 |
| 0.85 (operating point) | 0.914 | 0.614 | 0.734 |

**Deployment-distribution validation** (two random, non-confidence-sorted samples from unseen raw-crawl URLs scoring ≥ 0.85, manually checked):

| sample | domain | n | precision |
|---|---|---|---|
| 1 | lois.justice.gc.ca | 21 | 0.952 (20/21) |
| 2 | law.cornell.edu | 20 | 0.950 (19/20) |
| combined | both | 41 | 0.951 (39/41) |

**How the +500 batch was sourced.** False-negative analysis on the first-pass model (756 legal / 2018 non_legal, threshold 0.9) found 67/151 held-out legal misses concentrated in 8 root domains, split between thin-domain (too few training examples) and within-domain URL-shape-diversity gaps (e.g. cornell.edu was well-represented overall but misses concentrated in URL shapes underrepresented in training). That analysis directly targeted the error-driven sourcing pass (`src/sourcing/fetch_targeted_urls.py`) that produced the 493 additional labels merged in above, rather than blind re-sampling.

**Known gaps.** Retraining on the expanded set shifted the precision/recall curve right (0.9 now only yields 37% recall vs. 55.6% before), which is why the operating threshold moved to 0.85. The deployment-distribution validation above addresses the base-rate-sensitivity gap, but only for the two domains (`justice.gc.ca`, `cornell.edu`) that happened to dominate the confident-legal predictions in the pools sampled — it doesn't say anything about domains that never clear 0.85 at all. The 5 thin domains from the false-negative analysis (vermont.gov, judiciary.uk, wa.gov, utcourts.gov, wvlegislature.gov) still have near-zero recall even after the targeted sourcing pass added ~40-50 examples each; closing that gap needs more volume than a URL-only feature space could resolve at that scale, not a threshold change.

## Code

```
src/
  sourcing/    pull URLs from Common Crawl, build labeling batches
  intake/      merge raw labeling-agent output into clean label files
  classifier/  train + evaluate the TF-IDF/LR model
  validation/  deployment-distribution precision sampling
```

**`src/sourcing/`**
- `fetch_candidate_urls.py`: Samples raw URLs from a CC snapshot via Athena TABLESAMPLE, writes `data/candidates/raw_pool.jsonl`. Pure sourcing, no classification.
- `build_label_batch.py`: Turns that raw pool into the batch handed to the labeling agent — mixes rule-based prefilter hits (to boost legal density, since legal pages are ~0.2–0.3% of the raw crawl), raw random URLs, and synthetic homepage URLs for whitelisted domains (deliberate hard negatives for the index-vs-filing failure mode). Writes `data/candidates/candidates.jsonl`.
- `fetch_targeted_urls.py`: Error-driven follow-up pull, not random — targets the specific domains/URL-shapes the false-negative analysis flagged. Writes `data/candidates/targeted_batch.jsonl`.
- `fetch_raw_pool_no_ca.py`: Second Athena pull excluding `.ca` domains, used to source the second deployment-validation sample from a different slice of the crawl.

**`src/intake/`**
- `intake.py`: Merges the labeling agent's raw per-worker output into `data/processed/labeled_urls.jsonl`, applying the two documented label corrections (link-only pages, index/landing pages) found during audit.
- `intake_targeted.py`: Same merge pattern, scoped to the targeted +500 batch. Writes `data/processed/targeted_labeled_urls.jsonl`.

**`src/classifier/`**
- `train_classifier.py`: Trains the char n-gram TF-IDF + logistic regression classifier on both label files, reports held-out metrics and a threshold sweep.
- `threshold_sweep_full.py`: Full precision/recall/F1 sweep across thresholds, written to `data/processed/threshold_sweep_results.csv` for the record.

**`src/validation/`**
- `sample_deployment_validation.py` / `sample_deployment_validation_2.py`: Score the full (unlabeled) raw pool with the trained model, draw a genuinely random sample of the URLs that clear the operating threshold for hand-labeling — the deployment-distribution precision check.

### Archive: rule-based classifier (`archive/rule-based/`)

Superseded by the char n-gram TF-IDF + logistic regression approach (see URL Classifier above), kept for reference.

- `URL_Classifier.py`: URL-based legal/non-legal classifier. Two-layer architecture: curated domain whitelist (`wl_candidates.txt`) checked first, then strict keyword matching on the hostname only (path ignored to prevent false positives).
- `WL_Builder.py`: Discovers candidate legal domains from a CC snapshot via Athena. Queries `url_host_name` grouped by page count and writes results to `wl_candidates.txt` for manual triage.
- `wl_candidates.txt`: Triaged whitelist of primary legal source domains (courts, legislatures, statute repositories). One entry per line; suffix matching at runtime covers all subdomains.
- `CC_Classifier_Test.py`: Samples URLs from a CC snapshot via Athena TABLESAMPLE, classifies them, and prints positives tagged `[WL]` or `[KW]` plus a negative sample for manual precision/recall review.
- `cl_validation_results.txt`: External recall validation of the rule-based classifier against CourtListener bulk opinion data.

## Paper
University of Florida undergraduate research.

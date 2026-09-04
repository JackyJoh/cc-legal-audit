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

Superseded by the text classifier below as of 2026-09-03. The numbers in this section are the URL model's own record and predate both the definition-popup label correction and the move to page text, so they no longer describe the current pipeline. A fuller account of why this approach was dropped goes here later.

#### Rule-based approach (archived)

Curated domain whitelist + a conservative hostname-keyword fallback (details in `archive/rule-based/`), validated at 88.4% recall against CourtListener bulk data after one round of gap-filling.

**Why this was dropped.** Both layers are hardcoded human judgments about which domains/terms "count," not something learned from data. Closing a recall gap meant manually finding and adding a domain, a process that doesn't scale and has to be redone per snapshot or jurisdiction. More fundamentally, hostname-level matching can't distinguish page *type*: it treats a bare homepage or a case-listing index page the same as an actual statute or opinion page on the same domain, since that distinction lives in the URL path and content, not the hostname. And a binary keep/reject decision gives no confidence signal to trade precision against recall. This motivated the pivot to a model trained on the actual distribution of legal vs. non-legal URLs.

#### Current approach: char n-gram TF-IDF + logistic regression

1. **Ground-truth labeling pipeline.** Wrote a labeling task prompt (`prompts/legal_url_labeling_task.md`) that has a labeling agent fetch and read each candidate page and judge it against a strict LEGAL/NON_LEGAL definition, grounded in the page's own content, not URL heuristics. LEGAL requires the source's primary function to be producing/publishing formal legal documents *and* the specific page's own HTML to contain the actual text of a filing, statute, bill, regulation, or court opinion; a page that merely links out to that text (a landing/index page) is NON_LEGAL, since Common Crawl only captures a page's own HTML, never what it links to.
2. **Candidate sourcing.** Three separate passes so far, each targeting a different gap found in the one before it (see below for how each was sourced): the original 2990-URL batch (`fetch_candidate_urls.py` + `build_label_batch.py`, mixing rule-based prefilter hits, raw random URLs, and synthetic homepage negatives), an error-driven +500 batch (`fetch_targeted_urls.py`), and a CourtListener-derived host sample (`fetch_cl_hostnames.py` + `fetch_cl_urls.py`).
3. **Labeling runs**: parallel instances of the labeling agent worked each batch, producing 4044 labeled URLs total (1156 legal / 2888 non_legal) across the three passes, plus skips where the source blocked automated fetches with no accessible mirror. This is a single automated labeling policy (same model/prompt) applied at scale with human audit after the fact, not independent human annotators.
4. **Audit, correction, and merge** (`intake.py`): one script merges all three passes' raw per-worker output into `data/processed/labeled_urls.jsonl`, tagging each row with which pass produced it. Manual review of the original batch's legal-labeled set caught two systematic errors under the original (looser) definition (18 link-only pages and 16 index/landing pages templated across a URL family by one worker), corrected to non_legal, which is what drove tightening the labeling definition in step 1. A follow-up grep-by-rationale-keyword audit found no further drift.
5. **Training** (`train_classifier.py`): char n-gram (3–5) TF-IDF vectorizer + logistic regression, fit on the corrected label set with an 80/20 stratified held-out split (seed 42). URL string only as the feature: no page content, no hardcoded domain or keyword lists.
6. **Threshold selection**: chose an operating threshold of 0.85 on predicted legal-probability, prioritizing precision over recall. False positives pollute the small legal bucket the downstream topic-diversity analysis depends on; false negatives are reabsorbed into the much larger non-legal bucket where they're a negligible rounding error.

**Results** (held-out test set, 809 URLs: 231 legal / 578 non_legal; full sweep in `data/processed/threshold_sweep_results.csv`):

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.5 | 0.777 | 0.918 | 0.841 |
| 0.35 (F1-optimal) | 0.758 | 0.961 | 0.847 |
| 0.85 (operating point) | 0.906 | 0.541 | 0.678 |

**Domain generalization: the real limit of this classifier** (`eval_grouped.py`). The 1156 legal examples span 33 registered domains, but 3 of them (cornell.edu, justice.gc.ca, virginia.gov) account for 82.9% of all legal training data. That concentration matters because it's not just an imbalance problem:

- **Grouped split** (whole registered domains held out instead of random rows): recall collapses to 0 at any threshold ≥ 0.65, and only reaches 8.8% at 0.5.
- **Leave-one-domain-out**: retraining without each legal-bearing domain and scoring only that domain gives **0.000 recall for every one of the 16 domains tested**, including cornell.edu (422 examples) and justice.gc.ca (342 examples) when held out.
- **Coefficient audit**: only 5 of the top 30 positive-weight features are substrings of a training hostname; the rest are legal-vocabulary substrings (`sec`, `text`, `code`, `htm`). So this isn't crude hostname-memorization, it still fails to transfer to domains the model hasn't seen labeled examples from.

In short: this classifier recognizes legal *publishers* it has training data for, not legal text in general. It cannot currently identify a legal domain it has never seen a labeled example from, regardless of threshold.

**Tiered threshold (candidate mitigation, not yet deployment-validated).** Splitting the held-out test set by whether a URL's domain is one of the top-3 training domains shows the 0.85 operating threshold is badly miscalibrated for everything outside them:

| domain tier | test rows (legal) | threshold | precision | recall |
|---|---|---|---|---|
| top-3 | 257 (187) | 0.85 | 0.906 | 0.668 |
| top-3 | 257 (187) | 0.70 | 0.823 | 0.947 |
| non-top-3, in training | 214 (42) | 0.85 | 0 | 0 |
| non-top-3, in training | 214 (42) | 0.70 | 1.000 | 0.190 (n=8) |
| never seen in training | 338 (2) | any tested | 0 | 0 |

Dropping to ~0.70 for domains outside the top 3 recovers some recall on the *thin-but-known* tail (judiciary.uk, wa.gov, vermont.gov, etc.), at the cost of small sample sizes. It does nothing for domains with zero training examples, and the tier itself is a hardcoded domain lookup sitting on top of the model's score, not something the model decides, worth stating plainly given the project's own case against domain whitelisting for classification. Precision at these tiers hasn't been checked against fresh deployment-distribution sampling yet.

**Deployment-distribution validation** (two random, non-confidence-sorted samples from unseen raw-crawl URLs scoring ≥ 0.85, manually checked). Predates the CourtListener host-sample batch above and the resulting threshold/dataset changes, so treat as directional, not a current-model guarantee, until re-run:

| sample | domain | n | precision |
|---|---|---|---|
| 1 | lois.justice.gc.ca | 21 | 0.952 (20/21) |
| 2 | law.cornell.edu | 20 | 0.950 (19/20) |
| combined | both | 41 | 0.951 (39/41) |

**How the +500 batch was sourced.** False-negative analysis on the first-pass model (756 legal / 2018 non_legal, threshold 0.9) found 67/151 held-out legal misses concentrated in 8 root domains, split between thin-domain (too few training examples) and within-domain URL-shape-diversity gaps (e.g. cornell.edu was well-represented overall but misses concentrated in URL shapes underrepresented in training). That analysis directly targeted the error-driven sourcing pass (`fetch_targeted_urls.py`) that produced the 493 additional labels merged in above, rather than blind re-sampling.

**How the CourtListener host-sample batch was sourced.** Aimed squarely at the domain-generalization gap above: `fetch_cl_hostnames.py` builds a directory of court and legislature hostnames (CourtListener API + a closed enumeration of all 50 state legislature sites), `count_cl_captures.py` ranks them by actual Common Crawl depth, and `fetch_cl_urls.py` draws a per-publisher sample from Common Crawl itself (never from CourtListener's own URLs, to avoid a distribution shift). It added breadth, 777 labels across many publishers seen shallowly, but per the leave-one-domain-out results above, it did not fix the underlying generalization gap.

### Text classifier (bag of words on page text)

Status: initial result, recorded 2026-09-03. To be expanded later with a fuller write-up of why the URL-only approach was dropped.

Same 4044 labels, same evaluation script, same splits. The only thing that changes is what the model reads: the extracted page text from each URL's Common Crawl capture, instead of the URL string.

**Why.** The URL classifier could not recognize a legal publisher it had no training examples from. Leave-one-domain-out recall was 0.000 at every usable threshold. Page text has cross-publisher signal that URL strings do not: statutory prose from Kansas reads like statutory prose from Florida, while `ksrevisor.gov` and `flsenate.gov` share nothing as strings.

**Pipeline.**

1. `src/text/fetch_warc_text.py` joins the labeled URLs to the Common Crawl index through Athena to get each capture's WARC filename, byte offset and length, then range-fetches those bytes over HTTP and extracts text with trafilatura. 3996 of 4044 labeled URLs (98.8%) are present in CC-MAIN-2026-12, and 3787 (93.6%) produced usable text: 902 legal / 2885 non_legal. Per-publisher coverage is 92 to 100%, so no domain is too depleted to hold out fairly.
2. `src/classifier/features.py` holds both feature recipes in one place so training and evaluation cannot drift onto different settings. `url` is the existing char 3-5 gram vectorizer, unchanged. `text` is word unigrams and bigrams over the page body.
3. `eval_grouped.py --features url|text` runs identical splits and metrics against either, which is what makes the two comparable.

**Extraction settings** are load-bearing, so trafilatura is pinned at 2.2.0 and the version is recorded on every output row:

- `deduplicate=False`, set explicitly. The option strips repeated segments. An extractor that quietly dedupes its own input would confound the fuzzy dedup study this corpus exists to support.
- `include_tables=True`. Statutes and regulations are frequently laid out in tables.
- `favor_recall=True`. Dropping subsection (b)(2) is a worse error here than keeping an extra paragraph.

`data/processed/warc_pointers.jsonl` is committed and regenerates the text with no further Athena spend. The extracted text itself (about 25MB) is gitignored.

**Domain purity filter.** Any n-gram appearing in fewer than 3 distinct registered domains is dropped, counted on the training fold only so a held-out publisher can never influence the feature set. This removes site template text without a hand-written stoplist. Justice Canada page furniture (`marginal note`, `details date`, `date modified`, `page details`) sits at 1 to 2 domains, while real legal vocabulary starts at 17 (`general assembly`) and runs to 79 (`section`). k=3 is the smallest value that clears the template band, chosen from that gap rather than from a score: k of 1, 2, 3 and 5 all give leave-one-domain-out F1 between 0.794 and 0.801, so the choice is not performance-driven.

**Results, held-out test split** (80/20 random, seed 42; 758 rows, 181 legal / 577 non_legal):

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.50 | 0.771 | 0.912 | 0.835 |
| 0.60 (best F1) | 0.908 | 0.873 | 0.890 |
| 0.65 | 0.932 | 0.834 | 0.880 |
| 0.85 | 0.966 | 0.475 | 0.637 |

**Results, leave-one-domain-out** (macro over 16 held-out publishers), with the URL model's recall alongside for comparison:

| threshold | text precision | text recall | text F1 | URL recall |
|---|---|---|---|---|
| 0.50 | 0.901 | 0.825 | 0.846 | 0.204 |
| 0.60 | 0.923 | 0.739 | 0.801 | 0.072 |
| 0.65 | 0.932 | 0.696 | 0.779 | 0.011 |
| 0.85 | 0.987 | 0.421 | 0.556 | 0.000 |

Reading page text raises recall on publishers the model has never seen from effectively zero to 0.825 at t=0.50, holding 0.901 precision. The conclusion recorded above for the URL classifier, that it recognizes legal publishers rather than legal text, was correct for that model and does not hold for this one.

Leave-one-domain-out precision is also nearly flat from 0.50 to 0.85 (0.901 to 0.987) while recall more than halves. The precision-over-recall reasoning behind the URL model's 0.85 operating point does not carry over: the useful range here is 0.50 to 0.60.

**Known limits.**

- `cornell.edu` is the one weak publisher, at 0.494 precision (t=0.60) against 0.64 to 1.00 everywhere else. Leave-one-domain-out scores only the held-out publisher's own rows, so every false positive there is a Cornell non-legal page, and Cornell is the only publisher with a large body of labeled hard negatives (the definition popups). A 50-word popup and a 60-word statute section look alike in bag of words.
- The domain purity filter works on phrases but leaks on single words. `marginal`, `modified` and `note` survive because they appear incidentally on a few unrelated sites, while most of their weight still comes from one publisher. The fix is to filter on domain *concentration*, the share of a feature's occurrences coming from its top domain, rather than domain count.
- The positive class is still roughly 80% statutes and regulations. Court opinions and bills are barely represented, so the corpus stays register-narrow even though the model now generalizes across publishers.

## Code

```
src/
  sourcing/    pull URLs from Common Crawl, build labeling batches
  intake/      merge raw labeling-agent output into clean label files
  text/        fetch page text from Common Crawl for the labeled URLs
  classifier/  train + evaluate the TF-IDF/LR model
  validation/  deployment-distribution precision sampling
```

**`src/sourcing/`**
- `fetch_candidate_urls.py`: Samples raw URLs from a CC snapshot via Athena TABLESAMPLE, writes `data/candidates/raw_pool.jsonl`. Pure sourcing, no classification.
- `build_label_batch.py`: Turns that raw pool into the batch handed to the labeling agent — mixes rule-based prefilter hits (to boost legal density, since legal pages are ~0.2–0.3% of the raw crawl), raw random URLs, and synthetic homepage URLs for whitelisted domains (deliberate hard negatives for the index-vs-filing failure mode). Writes `data/candidates/candidates.jsonl`.
- `fetch_targeted_urls.py`: Error-driven follow-up pull, not random — targets the specific domains/URL-shapes the false-negative analysis flagged. Writes `data/candidates/targeted_batch.jsonl`.
- `fetch_raw_pool_no_ca.py`: Second Athena pull excluding `.ca` domains, used to source the second deployment-validation sample from a different slice of the crawl.
- `fetch_cl_hostnames.py`: Builds a directory of likely-legal hostnames from the CourtListener courts API plus a closed enumeration of all 50 state legislature sites. Writes `data/candidates/court_hostnames.jsonl`.
- `count_cl_captures.py`: Ranks those hostnames by actual Common Crawl capture depth, since CourtListener docket size and crawl coverage are uncorrelated. Writes `data/candidates/cc_host_counts.jsonl`.
- `fetch_cl_urls.py`: Draws a per-publisher URL sample from Common Crawl for the hostnames above (random URLs for hard negatives, one URL per path prefix for section coverage). Writes `data/candidates/host_sample_batch.jsonl`.
- `athena.py`: Shared Athena query-runner (used by `count_cl_captures.py` and `fetch_cl_urls.py`) that also reports bytes scanned and estimated cost.

**`src/intake/`**
- `intake.py`: Merges all three labeling passes' raw per-worker output into one file, `data/processed/labeled_urls.jsonl`, tagging each row with a `source` field (`original` / `target` / `cl`). Applies the two documented label corrections (link-only pages, index/landing pages) to the original pass only.

**`src/text/`**
- `fetch_warc_text.py`: Resolves each labeled URL to its Common Crawl capture (Athena, cached to `data/processed/warc_pointers.jsonl`), range-fetches the WARC record over HTTP, and extracts text with trafilatura. Writes `data/processed/labeled_text.jsonl`. Every input URL gets an output row, carrying a `skip_reason` when extraction failed, so counts always reconcile against the label file.

**`src/classifier/`**
- `features.py`: The two feature recipes, `url` (char 3-5 grams on the URL string) and `text` (word 1-2 grams on the page body), plus the domain purity filter. Kept in one file so training and evaluation cannot drift onto different settings.
- `train_classifier.py`: Trains the char n-gram TF-IDF + logistic regression classifier on the merged label file, reports held-out metrics and a threshold sweep.
- `threshold_sweep_full.py`: Full precision/recall/F1 sweep across thresholds, written to `data/processed/threshold_sweep_results.csv` for the record.
- `eval_grouped.py`: Diagnostic, not the shipped metrics. Takes `--features url|text` so both models are scored by identical code — compares a random row split against a grouped (whole-domain-held-out) split and a leave-one-domain-out sweep, plus a coefficient audit, to check whether the classifier generalizes to unseen legal publishers or just recognizes known ones.

**`src/validation/`**
- `sample_deployment_validation.py` / `sample_deployment_validation_2.py`: Score the full (unlabeled) raw pool with the trained model, draw a genuinely random sample of the URLs that clear the operating threshold for hand-labeling — the deployment-distribution precision check.
- `scan_domain_breakdown.py`: Scans the full raw pool at a given threshold and groups the flagged URLs by root domain, to see which publishers the model's confident predictions actually concentrate on.

### Archive: rule-based classifier (`archive/rule-based/`)

Superseded by the char n-gram TF-IDF + logistic regression approach (see URL Classifier above), kept for reference.

- `URL_Classifier.py`: URL-based legal/non-legal classifier. Two-layer architecture: curated domain whitelist (`wl_candidates.txt`) checked first, then strict keyword matching on the hostname only (path ignored to prevent false positives).
- `WL_Builder.py`: Discovers candidate legal domains from a CC snapshot via Athena. Queries `url_host_name` grouped by page count and writes results to `wl_candidates.txt` for manual triage.
- `wl_candidates.txt`: Triaged whitelist of primary legal source domains (courts, legislatures, statute repositories). One entry per line; suffix matching at runtime covers all subdomains.
- `CC_Classifier_Test.py`: Samples URLs from a CC snapshot via Athena TABLESAMPLE, classifies them, and prints positives tagged `[WL]` or `[KW]` plus a negative sample for manual precision/recall review.
- `cl_validation_results.txt`: External recall validation of the rule-based classifier against CourtListener bulk opinion data.

## Paper
University of Florida undergraduate research.

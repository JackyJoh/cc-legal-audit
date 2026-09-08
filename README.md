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

### URL Classifier (superseded 2026-09-03)

Kept as a record. Both approaches below were dropped; the numbers predate the definition-popup label correction and the move to page text.

**Rule-based (archived, `archive/rule-based/`).** Curated domain whitelist plus a hostname-keyword fallback, 88.4% recall against CourtListener bulk data. Dropped because both layers are hardcoded human judgments that have to be redone per snapshot, hostname matching cannot tell a homepage from a statute on the same domain, and a binary decision gives no confidence signal to trade precision against recall.

**char 3-5 gram TF-IDF + logistic regression.** Same labeling pipeline and label set as the text classifier below, reading the URL string instead of the page body. Held out 809 URLs: 0.906 precision / 0.541 recall at the 0.85 operating point, 0.777 / 0.918 at 0.50.

**Why it was dropped.** Leave-one-domain-out recall was **0.000 for all 16 tested publishers**, including cornell.edu (422 examples) and justice.gc.ca (342) when held out; grouped-split recall collapsed to 0 above 0.65. Only 5 of the top 30 features were hostname substrings, so this was not crude hostname memorization, and it still failed to transfer to any unseen publisher. The classifier recognized legal *publishers* it had data for, not legal text.

A tiered threshold (0.70 outside the top-3 domains) was drafted as a workaround and never validated. It was a hardcoded domain lookup sitting on the model's score, which is the same thing this project's case against whitelisting rejects, and the text classifier reaches 0.825 leave-one-domain-out recall without it.

Two error-driven sourcing passes came out of this work and their labels are still in use: an error-driven +500 batch (`fetch_targeted_urls.py`), sourced from a false-negative analysis that found 67/151 held-out misses concentrated in 8 root domains split between thin-domain and within-domain URL-shape gaps; and a 777-label CourtListener host sample (`fetch_cl_hostnames.py` + `fetch_cl_urls.py`) drawn from Common Crawl rather than from CourtListener's own URLs, aimed at the generalization gap. The host sample added breadth but did not close it.

### Text classifier (bag of words on page text)

Word 1-2 grams over the extracted page body instead of the URL string, same labels and same evaluation code. Page text carries cross-publisher signal that URL strings do not: statutory prose from Kansas reads like statutory prose from Florida, while `ksrevisor.gov` and `flsenate.gov` share nothing as strings.

**Leave-one-domain-out**, macro over 16 held-out publishers, with the URL model alongside (recorded 2026-09-03 on 4044 labels; later sections supersede):

| threshold | text precision | text recall | URL recall |
|---|---|---|---|
| 0.50 | 0.901 | 0.825 | 0.204 |
| 0.65 | 0.932 | 0.696 | 0.011 |
| 0.85 | 0.987 | 0.421 | 0.000 |

Recall on unseen publishers goes from effectively zero to 0.825 at 0.901 precision. LODO precision is nearly flat from 0.50 to 0.85 while recall halves, so the URL model's precision-first 0.85 does not carry over; the useful range is 0.50 to 0.60.

**Extraction is load-bearing**, so trafilatura is pinned at 2.2.0 and recorded on every row: `deduplicate=False` (an extractor that dedupes its own input would confound a fuzzy-dedup study), `include_tables=True` (statutes live in tables), `favor_recall=True`.

**Domain purity filter.** Any n-gram on fewer than 3 distinct registered domains is dropped, counted on the training fold only. Justice Canada page furniture (`marginal note`, `date modified`) sits at 1-2 domains while real legal vocabulary starts at 17 (`general assembly`), so k=3 is read off that gap rather than tuned: k of 1, 2, 3 and 5 all give LODO F1 between 0.794 and 0.801.

**Known limits.**

- `cornell.edu` is the weak publisher at 0.494 precision (t=0.60) against 0.64-1.00 elsewhere; a 50-word definition popup and a 60-word statute section look alike in bag of words.
- The purity filter works on phrases but leaks on single words: `marginal`, `modified`, `note` survive on incidental cross-domain use. Filtering on domain *concentration* rather than count would fix it.
- The positive class is register-narrow: 69% statute, 45% regulation (overlapping), 3.9% bills, 2.9% opinions. The 28 opinions sit almost entirely in cornell.edu (16) and judiciary.uk (10).
- Label set is ~23% legal; the crawl is 0.12%. Precision does not transfer across that gap, only recall and FPR do. Measured separately below.

### Bills batch (2026-09-07)

Bills were 3.9% of the legal class, so Open States supplied each legislature's bill section and Common Crawl supplied the URLs: 400 pages, 16 from each of 25 states.

375 labeled, **32 legal**. Only ~9% of pages under a legislature's own bill path are bill text; the rest are status pages and indexes linking to the text elsewhere. Label set went to 4419 URLs / 987 legal / 36 legal-bearing domains.

On the 14 publishers whose legal-row count did not change, macro LODO recall rose **0.419 to 0.436** (six up, eight held, none worse), all on publishers the batch never touched. Real, consistent, small for 400 labels.

### Publisher weighting

Three publishers hold 77% of the legal class. `domain_weights()` weights each row `(1 / its publisher's rows in that class) ** alpha`, rescaled to average 1 so regularisation strength does not move with alpha. Group imbalance, not leakage: LODO already holds publishers out.

| alpha | macro LODO recall (21 publishers, t=0.85) |
|---|---|
| 0 | 0.399 |
| **0.5 (default)** | **0.475** |
| 1.0 | 0.438 |

14 publishers improved, 7 held, none worse. The coefficient audit shows the mechanism: Justice Canada furniture demoted (`marginal` 6th to 25th, `note` and `details` out of the top 30), register vocabulary promoted (`section` 10th to 3rd). 1.0 overcorrects, letting a 5-row publisher outvote a 342-row one.

**Caveat.** Alpha was chosen by reading the metric it is reported on, so 0.475 is optimistically biased. Winner's curse, not leakage; nested selection would remove it and has not been run.

### Deployment precision, measured (2026-09-07)

First measurement on the real crawl distribution. 32,000 URLs drawn uniformly, every page fetched and scored, every flagged page kept and labeled. 152 labeled, 17 skipped: **27 legal, 125 non_legal.**

| threshold | flagged | legal | precision | 95% CI |
|---|---|---|---|---|
| 0.50 | 152 | 27 | 0.178 | [0.125, 0.246] |
| **0.60** | 93 | 25 | **0.269** | [0.189, 0.367] |
| 0.75 | 42 | 17 | 0.405 | [0.270, 0.555] |
| 0.85 | 18 | 11 | 0.611 | [0.386, 0.797] |
| 0.90 | 12 | 8 | 0.667 | [0.391, 0.862] |

**Base rate: 0.12%.** Publisher spread is healthy (143 flags at t=0.60 across 74 domains, against the URL model's 24 flags all on justice.gc.ca), so the model ranks correctly and the decision boundary is what sits wrong. No threshold is usable: 0.90 reaches 0.667 and flags 12 pages in 32,000.

The false positives are one class, legal-sounding text written by a private party: an SEC contract exhibit at 0.976, tax-treaty commentary at 0.960, purchase-order terms at 0.957, a news article about a city ordinance at 0.923. No training negative taught that distinction, because every earlier batch sampled from publishers of law.

### Retraining on the mined negatives

The 152 labels went back into training, one retrain by design (4,490 rows, 1,014 legal). Macro LODO recall fell **0.399 to 0.375** on the 21 comparable publishers (eight down, thirteen held, none up); grouped-split precision rose at low thresholds (0.50: 0.457 to 0.492).

That is hard negatives working, buying precision with recall, and it was not enough. Rescoring the same 152 pages gives 0.711 at t=0.60, but they are in training now, so that is a memorisation ceiling. The model still scores the tax-treaty site at 0.854 and the SEC exhibit at 0.829 with both in training, which says the class is not linearly separable from primary law here.

### Decision: source the legal bucket by publisher, classify page type within it

Open-web detection fails on the base rate, not the model. Solving `precision = (base x recall) / (base x recall + (1-base) x FPR)` at base 0.12%, recall 0.5:

| target precision | required FPR | vs. measured 0.227% |
|---|---|---|
| 80% | 0.015% | 15x reduction |
| 90% | 0.0067% | 34x reduction |

No feasible amount of hard-negative mining delivers a 15x cut. Restricting the pool to authority-enumerated legal publishers removes the problem instead:

| | open-web pool | authority-domain pool |
|---|---|---|
| base rate | 0.12% | **23.3%** |
| precision @0.75 | 0.405 (measured) | 0.937 (held-out split) |
| precision @0.85 | 0.611 (measured) | 0.964 (held-out split) |

The authority pool's base rate matches the label set's, so held-out random-split numbers transfer with no base-rate translation, the step that destroyed every open-web projection. Only 119 of 32,000 uniformly drawn pages (0.37%) sit on an authority domain: the filter discards 99.6% of the crawl and raises the base rate ~200-fold.

**Not the whitelisting this project rejected.** That objection was to a whitelist as a *classifier*, since a hostname cannot tell a statute from a docket index, and it stands. Enumerated publishers *source* the corpus; the classifier still does what the host list cannot, since only ~9% of pages inside a legislature's bill section are bill text.

**Both buckets stay Common Crawl.** The authority supplies hostnames only, never URLs or documents, exactly as `fetch_cl_urls.py` and `fetch_bill_urls.py` already work. The general-web bucket should be drawn by the same host-level procedure over random hosts so both sides are defined the same way.

**Cost.** The corpus becomes "legal text from known legal publishers," not "all legal text in Common Crawl," giving up claims about crawl-wide legal share, web-wide representativeness, and legal text off publisher domains. Composition becomes a choice rather than a consequence of uniform sampling, so the unsourced opinion register matters more and entropy should be reported per register.

### Sourcing the missing registers

An outside authority says where each publisher files documents; Common Crawl supplies the actual URLs, so nothing is sampled from the authority and there is no distribution shift. `fetch_register_sources.py` does this step.

**Bills: works.** Open States covers all 50 state legislatures and records a `sources` link back to each bill's page on its own state site.

**Court opinions: this route does not work.** Over 220 CourtListener opinion records, 96% carry no `download_url`. Its corpus is bulk donations (`html_columbia`, `xml_harvard`, `html_lawbox`), so it holds opinion text but no record of the page it came from. The 9 records with an address gave a govinfo PDF directory, an upload folder and PACER's paywalled gateway. That is a property of the source, not the sample size, so opinions need a different authority.

PDFs remain unsettled as *documents*: a second extractor is needed, and PDF-to-text differs from HTML-to-text in hyphenation and page furniture, which is exactly the surface variation 13-gram MinHash responds to. Doing it on the legal side only would confound the headline comparison.

### The authority domain list

`court_hostnames.jsonl` holds **342 hostnames across 120 registered domains**: 292 courts recovered from the CourtListener API page cache plus a closed enumeration of all 50 state legislature sites. `build_legal_domains.py` merges these with the Open States bill sections into `legal_domains.jsonl`, **347 domains**, each carrying the sampling unit correct for it.

Courts and legislatures need different units. A state legislature is its own registrant, so `codes.ohio.gov` and `www.legislature.ohio.gov` are one publisher and merge at the registered-domain level. Federal courts are the opposite: 78 of the CourtListener hostnames share `uscourts.gov`, so collapsing by registered domain would merge 86 separate courts into one entry. Courts stay at hostname granularity, legislatures merge at registered domain.

CourtListener's `url` field is wrong often enough to need a denylist: it points the NY Surrogate's Court at Wikipedia, the Emergency Court of Appeals at Wikimedia, a bankruptcy court at the FJC's own site, and a defunct Tennessee court at the genealogy service FamilySearch, which alone carried enough captures to rank in the top 40 legal domains.

`count_domain_captures.py` measures real crawl depth per domain: **286 of 347 domains appear in CC-MAIN-2026-12, 614,044 eligible pages**. The 61 with zero captures are almost all state courts, split between dead legacy `state.XX.us` hosts and live sites that block Common Crawl in robots.txt.

### Deployment precision, final (2026-09-08) — classifier closed

250 pages drawn from inside the legal domains and hand-labeled, in three score strata so the band above 0.85 gets a usable read regardless of how the pool skews. Each label is weighted by its stratum's frame size; intervals are a stratified bootstrap. `precision_report.py` produces all of it.

| t | precision | contamination | recall | yield |
|---|---|---|---|---|
| 0.60 | 0.911 | 8.9% | 0.770 | 30.6% |
| **0.75** | **0.953** | **4.7%** | **0.602** | **22.9%** |
| 0.85 | 0.980 | 2.0% | 0.418 | 15.4% |

**Base rate inside these domains is 36.2%**, against 0.12% on the open web, so the pivot changed the problem by roughly 300x. Measured precision beats the grouped-split estimate (0.900 at 0.85) because deployment is in-domain while the grouped split holds whole publishers out.

**Operating threshold: 0.75.** At scale that is 140,405 pages kept, **133,859 legal documents**, 6,546 false positives, and a false-positive rate of 1.67% of the 391,846 non-legal pages. Contamination is the number for corpus quality, FPR the number for filter behaviour.

**The misses are short documents, not a random slice.** Legal pages at or above 0.85 run a median 1,108 words with 3% under 200; legal pages below it run a median 238 words with 37% under 200, the same length profile as the non-legal pages. At the margin this is partly a length detector, the short-document failure first seen on `cornell.edu`. Short and long documents carry different near-duplicate structure, so a threshold that preferentially drops short legal documents biases the corpus toward long ones, and that lands on ΔH. It gets worse as the threshold rises, which is a second argument for 0.75 over 0.85, and it belongs in limitations as a real confound.

**Open questions.**

- Opinions remain unsourced, and matter more under this plan.
- `OPERATING_THRESHOLD` in `eval_grouped.py` is hardcoded to 0.85, so its LODO figures are measured at a threshold the model does not run at. Before/after comparisons are unaffected.
- `register_sources_bills.jsonl` was drawn at `--pages 3`, 60 bills per state.
- Corpus construction, sampling and the dedup sweep design are the next phase.

## Code

```
src/
  samples/     draw URL samples from Common Crawl, build labeling batches
  labels/      merge raw labeling-agent output into clean label files
  corpus/      fetch and extract page text from Common Crawl
  classifier/  train, score and evaluate the TF-IDF/LR model
  validation/  deployment-distribution precision sampling
```

**`src/samples/`**
- `fetch_candidate_urls.py`: Uniform random URL sample from a snapshot. Counts the population first, then `TABLESAMPLE BERNOULLI(p)` with no `LIMIT`, since a `LIMIT` returns whichever index files answered first and shuffling cannot repair that. Writes `raw_pool.jsonl`.
- `build_label_batch.py`: Mixes prefilter hits, random URLs and synthetic homepage negatives into `candidates.jsonl`.
- `fetch_targeted_urls.py`: Error-driven pull at the domains and URL shapes the false-negative analysis flagged.
- `fetch_raw_pool_no_ca.py`: Second pull excluding `.ca`, for a validation sample from a different slice of the crawl.
- `fetch_cl_hostnames.py`: Court hostnames from the CourtListener API plus all 50 state legislature sites. Pages are cached to disk, so a run that hits the hourly cap or crashes resumes free. Writes `court_hostnames.jsonl`.
- `build_legal_domains.py`: Merges the court and legislature host lists into `legal_domains.jsonl`, each row carrying the sampling unit correct for it, host for courts and registered domain for legislatures.
- `count_domain_captures.py`: Ranks legal domains (courts and legislatures, from `legal_domains.jsonl`) by actual crawl depth, since docket/bill-record size and crawl coverage are uncorrelated.
- `fetch_legal_pool.py`: Uniform random sample of eligible pages across the legal domains, the pool the precision sample is drawn from. Ordered by seeded hash, so a larger `--n` is a superset of a smaller one.
- `fetch_precision_sample.py`: The hand-labeling batch, drawn as three score strata so the band above 0.85 gets a usable read whatever the pool's skew. Batch, scores and per-stratum frame sizes go to separate files; the labeling agent sees only the batch.
- `fetch_cl_urls.py`: Per-publisher URL sample from Common Crawl for those hostnames. Writes `host_sample_batch.jsonl`.
- `fetch_register_sources.py`: Asks Open States where each of the 50 states files its bills, reduced to site plus wildcard path (`/li/%/measures/%`). Touches neither Common Crawl nor the documents.
- `fetch_bill_urls.py`: Turns those sections into a batch. One Athena query sorts every captured page `in` or `out` by path match, drops sites under 200 captured pages, takes a fixed quota each so a deeply-crawled state cannot dominate.
- `fetch_flagged_urls.py`: Uniform crawl sample, scored, keeping what the model flags. The only sampler that does not draw from where law is expected to be, which is what makes its negatives representative of real failures. URLs and scores go to separate files so the labeling agent cannot see confidence.
- `athena.py`: Shared query runner, reports bytes scanned and estimated cost.

**`src/labels/`**
- `intake.py`: Merges all five labeling passes into `labeled_urls.jsonl`, tagging each row with its `source`, and applies the documented label corrections (link-only pages, index/landing pages, definition popups) per pass.

**`src/corpus/`**
- `fetch_warc_text.py`: Resolves each URL to its capture (Athena, cached to `warc_pointers.jsonl`), range-fetches the WARC record over HTTP, extracts with trafilatura. `--input/--output` so the label set and any crawl sample go through identical extraction. Every input URL gets a row, carrying `skip_reason` on failure, so counts reconcile.

**`src/classifier/`**
- `features.py`: Both feature recipes, the domain purity filter, publisher weighting and label loading, in one file so train/score/eval cannot drift apart.
- `train_classifier.py`: `--features url|text`, `--domain-weight` (default 0.5). Reports held-out metrics, refits on all labels, writes `models/<mode>_clf.joblib` with provenance (label hash, extractor, versions).
- `score.py`: Scores any jsonl with a saved bundle, reports flag rate per threshold. Refuses on extractor mismatch.
- `threshold_sweep_full.py`: Full sweep 0.05-0.95 with tp/fp/fn to CSV.
- `eval_grouped.py`: Diagnostic, not shipped metrics. Random split vs grouped (whole domains held out) vs leave-one-domain-out, plus a coefficient audit. `--exclude` drops a batch before training, so running with and without it measures what that batch changed under identical code.
- `precision_report.py`: The shipped metric. Joins the hand labels to their scores, weights each label by its stratum's frame size, and reports precision, contamination, recall and yield by threshold with bootstrap intervals, then projects them onto every eligible page in the crawl.

**`src/validation/`**
- `sample_deployment_validation.py`: Scores a pool, draws a uniform random sample of what clears the threshold for hand-labeling. Random rather than confidence-sorted, since the top of the ranking would flatter the estimate.
- `scan_domain_breakdown.py`: Groups flagged pages by registered domain. A model can hold good precision while firing on only two or three sites.

Both load a saved model rather than fitting one, so a reported number always names the model that produced it.

### Archive: rule-based classifier (`archive/rule-based/`)

Superseded, kept for reference. `URL_Classifier.py` (whitelist then hostname keyword match), `WL_Builder.py` (discovers candidate domains via Athena for manual triage), `wl_candidates.txt`, `CC_Classifier_Test.py` (samples and classifies for manual review), `cl_validation_results.txt` (recall validation against CourtListener bulk data).

## Paper
University of Florida undergraduate research.

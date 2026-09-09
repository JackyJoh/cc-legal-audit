# Legal-document classifier

Sources the legal and general-web buckets used by the audit (see the [top-level README](../../README.md)) and classifies page type within the legal bucket. Closed as of 2026-09-08 at an operating threshold of 0.75 (0.953 precision, measured). Kept here as the design record: why the rule-based and URL-only approaches were dropped, why sourcing pivoted from open-web classification to publisher enumeration, and every measurement behind the final threshold.

**What shipped:** TF-IDF over word 1-2 grams of the extracted page body, logistic regression, scoring only pages that already sit on an authority-enumerated legal publisher. Two earlier approaches were tried and dropped; they are written up under [Approaches that were dropped](#approaches-that-were-dropped) rather than here, so this document reads forward from what works.

## Pipeline, in order

The order things actually ran in, which is not the order the final design would suggest. [Code](#code) describes what each script does; this is only the sequence and what each stage hands to the next. Paths are relative to the repo root.

The thing to notice: **publisher enumeration arrives in the middle, not at the start.** The first two label batches were sourced by the rule-based whitelist that this project later dropped. Authority enumeration was built afterwards, and it is what the last two batches and the entire final measurement rest on.

**1. Bootstrap the first batch with the rule-based prefilter.** `fetch_candidate_urls.py` → `raw_pool.jsonl` (uniform random from the snapshot), then `build_label_batch.py` → `candidates.jsonl`, mixing prefilter hits, raw random URLs and synthetic homepage negatives. The prefilter is the archived classifier: `build_label_batch.py` imports `URL_Classifier` from `archive/rule-based/` and reads its `wl_candidates.txt`. That archive is a live dependency of this step, not just a historical record.

**2. Expand on the errors.** `fetch_targeted_urls.py` → `targeted_batch.jsonl`, aimed at the false-negative domains and URL shapes found after training on batch 1.

**3. Enumerate the publishers.** Only now. `fetch_cl_hostnames.py` → `court_hostnames.jsonl` (CourtListener API, page-cached so a capped run resumes free) and `fetch_register_sources.py` → `register_sources_bills.jsonl` (Open States). `build_legal_domains.py` merges both into `legal_domains.jsonl`, then `count_domain_captures.py` measures real crawl depth per domain into `cc_domain_counts.jsonl`.

**4. Draw the batches that depend on the enumeration.** `fetch_cl_urls.py` → `host_sample_batch.jsonl` (court and legislature hostnames) and `fetch_bill_urls.py` → `bill_sample_batch.jsonl`, which reads `register_sources_bills.jsonl` from step 3. Both exclude URLs already claimed by earlier batches.

**5. Label, then merge.** Labeling output lands in `data/labels/run_*.jsonl`. `labels/intake.py` merges every pass available so far into `data/processed/labeled_urls.jsonl`, tagging each row with its pass. A labeled row whose URL is not in its own batch file is dropped as contamination, a duplicate label is deduped, a *conflicting* duplicate is logged rather than silently picked, and a label beats a skip.

Three corrections are applied at this step, all flipping `legal` → `non_legal`:

| correction | rows | what went wrong |
|---|---|---|
| link-only | 18 | The labeling prompt originally let a page that "directly links to" a filing count as legal. Common Crawl captures only a page's own HTML, so a landing page with no document text on it is a false positive. |
| index/landing | 16 | One worker applied a templated rationale across a whole Justice Laws Canada URL family without checking each page. Confirmed by refetching 5 of them. |
| definition popup | 311 | Cornell LII definition popups, short enough to look like a statute section in bag of words. |

They are corrected rather than discarded, because they are useful hard negatives: right domain, right vocabulary, wrong page type. The 311-row popup correction is the largest single change to the label set and predates every number in [Approaches that were dropped](#approaches-that-were-dropped).

**6. Fetch page text.** `../common/fetch_warc_text.py` locates each labeled URL in the crawl and extracts it, writing `warc_pointers.jsonl` (committed, small) and `labeled_text.jsonl` (gitignored, ~100MB, regenerates from the pointers with no further Athena spend).

**7. Train.** `model/train_classifier.py --features text` → `models/text_clf.joblib` (gitignored, ~5s to rebuild).

At this point there is a working model, and the last two stages are the ones that need one.

**8. Mine hard negatives from the open web.** `fetch_flagged_urls.py` scores a uniform crawl draw with the step 7 model and keeps what it flags → `flagged_sample_batch.jsonl`, plus a separate scores file so the labeling agent never sees confidence. These labels are the fifth pass, so they re-enter at step 5: **steps 5 → 6 → 7 run a second time** with the flagged batch included, and that retrained model is the one used below. This is the only loop in the pipeline.

**9. Measure deployment precision.** `fetch_legal_pool.py` reads `legal_domains.jsonl` from step 3 → `legal_pool.jsonl`, then `fetch_precision_sample.py` scores that pool with the retrained model and draws the 250-row batch, its scores, and per-stratum frame sizes into three separate files. Hand-label the batch, then `validation/precision_report.py` produces every number in [Deployment precision, final](#deployment-precision-final-2026-09-08--classifier-closed).

So the full run is **1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → (5 → 6 → 7 again) → 9.** Every step runs in its listed order; steps 5-7 simply run twice, because the flagged batch cannot exist until a model does.

Where the five label passes come from:

| pass | batch file | sourced by | lap |
|---|---|---|---|
| `original` | `candidates.jsonl` | archived rule-based whitelist (step 1) | first |
| `target` | `targeted_batch.jsonl` | error analysis on batch 1 (step 2) | first |
| `cl` | `host_sample_batch.jsonl` | authority enumeration (step 4) | first |
| `bills` | `bill_sample_batch.jsonl` | authority enumeration (step 4) | first |
| `flagged` | `flagged_sample_batch.jsonl` | a trained model on a uniform crawl draw (step 8) | second |

**One input has no producer in this tree.** `fetch_cl_urls.py` reads `data/candidates/cc_host_counts.jsonl`, which nothing here writes. It is a host-level counts file from an earlier version of `count_domain_captures.py`, which now rolls up to registered domains and writes `cc_domain_counts.jsonl` instead. The committed file is enough to rerun the step, but it cannot currently be regenerated from scratch.

## Text classifier (bag of words on page text)

This is the shipped model and the one used for the rest of this document: word 1-2 grams over the extracted page body, rather than over the URL string as the [superseded URL classifier](#url-classifier-superseded-2026-09-03) did, on the same labels and the same evaluation code. Page text carries cross-publisher signal that URL strings do not: statutory prose from Kansas reads like statutory prose from Florida, while `ksrevisor.gov` and `flsenate.gov` share nothing as strings.

**The model, exactly.** `features.py` holds the whole recipe, so the trainer and both evaluators cannot drift onto different settings.

- *Vectorizer:* TF-IDF over word 1-2 grams, `min_df=3`, `max_features=200_000`, `sublinear_tf=True`. No stop-word list, deliberately: dropping `to` turns `pursuant to` into `pursuant` and discards the collocation bigrams exist to catch, and idf already handles genuinely uninformative words. Trigrams are left out because at ~4k documents they are mostly singletons that `min_df` prunes anyway.
- *Estimator:* logistic regression, `class_weight="balanced"`, `max_iter=2000`. Balancing is load-bearing rather than cosmetic: legal is about a quarter of the labels and a fifth of the domains, and without it the fit optimises the majority class and the threshold sweep has nothing left to trade off.
- *Two fits per run.* `train_classifier.py` fits first on an 80/20 stratified split to produce the held-out metrics and threshold sweep — those are the numbers worth quoting — then refits on every label, and that second fit is what ships. The saved model has therefore seen every row, so its own scores on the label set are worthless as a measurement; the split fit exists to supply the measurement instead.
- *Provenance.* The bundle records the label file and its SHA-256, the extractor that produced the training text, the seed, the feature count after the purity filter, and the sklearn/Python versions. `score.py` refuses to run when the documents it is handed disagree with the vectorizer about any of it, so a reported number always names the model that produced it.

**Initial measurement, not final performance.** Recorded 2026-09-03 on 4044 labels, before the bills batch, publisher weighting, and deployment-precision retrain below. It's here to show the text classifier beats the URL classifier at generalizing to unseen publishers; the final numbers are in "Deployment precision, final" further down.

**Leave-one-domain-out**, macro over 16 held-out publishers, with the URL model alongside:

| threshold | text precision | text recall | URL recall |
|---|---|---|---|
| 0.50 | 0.901 | 0.825 | 0.204 |
| 0.65 | 0.932 | 0.696 | 0.011 |
| 0.85 | 0.987 | 0.421 | 0.000 |

Recall on unseen publishers goes from effectively zero to 0.825 at 0.901 precision. LODO precision is nearly flat from 0.50 to 0.85 while recall halves, so the URL model's precision-first 0.85 does not carry over; the useful range is 0.50 to 0.60.

**Extraction is load-bearing**, so trafilatura is pinned at 2.2.0 and recorded on every row: `deduplicate=False` (an extractor that dedupes its own input would confound a fuzzy-dedup study), `include_tables=True` (statutes live in tables), `favor_recall=True`.

**Domain purity filter.** Any n-gram on fewer than 3 distinct registered domains is dropped, counted on the training fold only. Justice Canada page furniture (`marginal note`, `date modified`) sits at 1-2 domains while real legal vocabulary starts at 17 (`general assembly`), so k=3 is read off that gap rather than tuned: k of 1, 2, 3 and 5 all give LODO F1 between 0.794 and 0.801. 3 is the smallest value that clears all four template phrases. 4 is missing from that list because it is ruled out on vocabulary retention rather than on F1, dropping it from 72% to 49% (recorded in `features.py`).

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

Three publishers hold 75% of the legal class. `domain_weights()` weights each row `(1 / its publisher's rows in that class) ** alpha`, rescaled to average 1 so regularisation strength does not move with alpha. Group imbalance, not leakage: LODO already holds publishers out.

| alpha | macro LODO recall (21 publishers, t=0.85) |
|---|---|
| 0 | 0.399 |
| **0.5 (default)** | **0.475** |
| 1.0 | 0.438 |

14 publishers improved, 7 held, none worse. The coefficient audit shows the mechanism: Justice Canada furniture demoted (`marginal` 6th to 25th, `note` and `details` out of the top 30), register vocabulary promoted (`section` 10th to 3rd). 1.0 overcorrects, letting a 5-row publisher outvote a 342-row one.

**Caveat.** Alpha was chosen by reading the metric it is reported on, so 0.475 is optimistically biased. Winner's curse, not leakage; nested selection would remove it and has not been run.

## Sourcing: why the legal bucket comes from publishers, not open-web detection

The four sections below are one argument in order: the open-web approach was measured, an attempted fix failed, the base-rate math explained why, and the pivot followed from it.

### Testing the open-web approach: precision measured, then rejected (2026-09-07)

Not the shipped configuration. This tests what happens if the classifier scores pages drawn uniformly from the whole crawl, before the authority-domain pivot below. 32,000 URLs drawn uniformly, every page fetched and scored, every flagged page kept and labeled. 152 labeled, 17 skipped: **27 legal, 125 non_legal.**

| threshold | flagged | legal | precision | 95% CI |
|---|---|---|---|---|
| 0.50 | 152 | 27 | 0.178 | [0.125, 0.246] |
| **0.60** | 93 | 25 | **0.269** | [0.189, 0.367] |
| 0.75 | 42 | 17 | 0.405 | [0.270, 0.555] |
| 0.85 | 18 | 11 | 0.611 | [0.386, 0.797] |
| 0.90 | 12 | 8 | 0.667 | [0.391, 0.862] |

**Verdict: no threshold is usable.** Even the strictest, 0.90, only reaches 0.667 precision, and flags just 12 pages out of 32,000. Publisher spread is healthy (143 flags at t=0.60 across 74 domains, against the URL model's 24 flags all on justice.gc.ca), so the model ranks correctly, it's the decision boundary that sits wrong at this base rate (legal pages are 0.12% of the crawl). This result is what motivated the pivot to authority-domain sourcing below.

The false positives are one class, legal-sounding text written by a private party: an SEC contract exhibit at 0.976, tax-treaty commentary at 0.960, purchase-order terms at 0.957, a news article about a city ordinance at 0.923. No training negative taught that distinction, because every earlier batch sampled from publishers of law.

### Attempted fix: retraining on the mined negatives (still open-web, still rejected)

The obvious next step before abandoning the open-web approach: fold those 152 hand labels back into training as hard negatives. One retrain by design (4,490 rows, 1,014 legal). Macro LODO recall fell **0.399 to 0.375** on the 21 comparable publishers (eight down, thirteen held, none up); grouped-split precision rose at low thresholds (0.50: 0.457 to 0.492).

That is hard negatives working, buying precision with recall, and it was not enough. Rescoring the same 152 pages gives 0.711 at t=0.60, but they are in training now, so that is a memorisation ceiling, not a working boundary. The model still scores the tax-treaty site at 0.854 and the SEC exhibit at 0.829 with both in training, which says the class is not linearly separable from primary law here. This confirms retraining alone can't save the open-web approach; the base-rate math below shows why.

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

Courts and legislatures need different units. A state legislature is its own registrant, so `codes.ohio.gov` and `www.legislature.ohio.gov` are one publisher and merge at the registered-domain level. Federal courts are the opposite: 199 of the CourtListener hostnames share `uscourts.gov`, so collapsing by registered domain would merge 222 separate courts into one entry. Courts stay at hostname granularity, legislatures merge at registered domain.

CourtListener's `url` field is wrong often enough to need a denylist: it points the NY Surrogate's Court at Wikipedia, the Emergency Court of Appeals at Wikimedia, a bankruptcy court at the FJC's own site, and a defunct Tennessee court at the genealogy service FamilySearch, which alone carried enough captures to rank in the top 40 legal domains.

`count_domain_captures.py` measures real crawl depth per domain: **286 of 347 domains appear in CC-MAIN-2026-12, 285 of them with eligible pages, 614,044 in total**. The 61 with zero captures are almost all state courts, split between dead legacy `state.XX.us` hosts and live sites that block Common Crawl in robots.txt.

## Deployment precision, final (2026-09-08) — classifier closed

250 pages drawn from inside the legal domains and hand-labeled, in three score strata so the band above 0.85 gets a usable read regardless of how the pool skews. Each label is weighted by its stratum's frame size; intervals are a stratified bootstrap. `precision_report.py` produces all of it.

| t | precision | 95% CI | contamination | recall | 95% CI | yield |
|---|---|---|---|---|---|---|
| 0.60 | 0.911 | [0.871, 0.946] | 8.9% | 0.770 | [0.645, 0.910] | 30.6% |
| **0.75** | **0.953** | **[0.918, 0.986]** | **4.7%** | **0.602** | **[0.502, 0.722]** | **22.9%** |
| 0.85 | 0.980 | [0.950, 1.000] | 2.0% | 0.418 | [0.351, 0.497] | 15.4% |

**Recall is the soft number here, precision the hard one.** The recall interval is about three times wider at every threshold, and structurally so rather than by bad luck: precision is decided by the pages the filter keeps, which sit in the two strata carrying 100 labels each, while recall is measured against the estimated legal count for the whole frame. 23% of that estimate (578 of 2,509 pages) comes from the 6 legal pages found in the low stratum, where one label stands for 96 pool pages. The same 6 rows set the 36.2% base rate below. Quote precision freely; treat recall as a band, not a point.

**Base rate inside these domains is 36.2%**, against 0.12% on the open web, so the pivot changed the problem by roughly 300x. Measured precision beats the grouped-split estimate (0.900 at 0.85) because deployment is in-domain while the grouped split holds whole publishers out.

**Operating threshold: 0.75.** At scale that is 140,405 pages kept, **133,859 legal documents**, 6,546 false positives, and a false-positive rate of 1.67% of the 391,846 non-legal pages. Contamination is the number for corpus quality, FPR the number for filter behaviour.

**The misses are short documents, not a random slice.** Legal pages at or above 0.85 run a median 1,108 words with 3% under 200; legal pages below it run a median 238 words with 37% under 200, the same length profile as the non-legal pages. At the margin this is partly a length detector, the short-document failure first seen on `cornell.edu`. Short and long documents carry different near-duplicate structure, so a threshold that preferentially drops short legal documents biases the corpus toward long ones, and that lands on ΔH. It gets worse as the threshold rises, which is a second argument for 0.75 over 0.85, and it belongs in limitations as a real confound.

**Open questions.**

- Opinions remain unsourced, and matter more under this plan.
- `OPERATING_THRESHOLD` in `model/eval_grouped.py` is hardcoded to 0.85, so its LODO figures are measured at a threshold the model does not run at. Before/after comparisons are unaffected.
- `register_sources_bills.jsonl` was drawn at `--pages 3`, 60 bills per state.
- Corpus construction, sampling and the dedup sweep design are the next phase (tracked at the [top level](../../README.md)).

## Approaches that were dropped

Kept as a record, and placed after the shipped design rather than before it. Both predate the definition-popup label correction and the move to page text, so their numbers are not comparable to the ones above.

### Rule-based (archived, `archive/rule-based/`)

Curated domain whitelist plus a hostname-keyword fallback, 88.4% recall against CourtListener bulk data. Dropped because both layers are hardcoded human judgments that have to be redone per snapshot, hostname matching cannot tell a homepage from a statute on the same domain, and a binary decision gives no confidence signal to trade precision against recall.

**Dropped as a classifier, still live as a sampler.** `build_label_batch.py` imports `URL_Classifier` from this directory and reads `wl_candidates.txt` to build the `original` batch, so the archive is a runtime dependency of [step 1](#pipeline-in-order) rather than a dead reference. Nothing it decides reaches a label: the prefilter only raises legal density in the batch handed to the labeling agent, and every URL is still confirmed or rejected by hand.

### URL classifier (superseded 2026-09-03)

Char 3-5 gram TF-IDF plus logistic regression: same labeling pipeline and label set as the text classifier, reading the URL string instead of the page body. Held out 809 URLs: 0.906 precision / 0.541 recall at the 0.85 operating point, 0.777 / 0.918 at 0.50.

**Why it was dropped.** Leave-one-domain-out recall was **0.000 for all 16 tested publishers**, including cornell.edu (422 examples) and justice.gc.ca (342) when held out; grouped-split recall collapsed to 0 above 0.65. Only 5 of the top 30 features were hostname substrings, so this was not crude hostname memorization, and it still failed to transfer to any unseen publisher. The classifier recognized legal *publishers* it had data for, not legal text.

**What survived it.** Two error-driven sourcing passes came out of this work and their labels are still in use: an error-driven +500 batch (`fetch_targeted_urls.py`), sourced from a false-negative analysis that found 67/151 held-out misses concentrated in 8 root domains split between thin-domain and within-domain URL-shape gaps; and a 777-label CourtListener host sample (`fetch_cl_hostnames.py` + `fetch_cl_urls.py`) drawn from Common Crawl rather than from CourtListener's own URLs, aimed at the generalization gap. The host sample added breadth but did not close it.

## Code

```
samples/     draw URL samples from Common Crawl, build labeling batches
labels/      merge raw labeling-agent output into clean label files
model/       train, score and evaluate the TF-IDF/LR model
validation/  score/sample the deployment distribution and report precision
archive/     superseded rule-based classifier, still imported by build_label_batch.py
```

Shared Common Crawl access (`athena.py`, WARC fetch/extract) lives in `../common/`, not here, since other pipeline stages need it too.

**`samples/`**
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

**`labels/`**
- `intake.py`: Merges all five labeling passes into `labeled_urls.jsonl`, tagging each row with its `source`, and applies the documented label corrections (link-only pages, index/landing pages, definition popups) per pass.

**`model/`**
- `features.py`: Both feature recipes, the domain purity filter, publisher weighting and label loading, in one file so train/score/eval cannot drift apart.
- `train_classifier.py`: `--features url|text`, `--domain-weight` (default 0.5). Reports held-out metrics, refits on all labels, writes `models/<mode>_clf.joblib` with provenance (label hash, extractor, versions).
- `score.py`: Scores any jsonl with a saved bundle, reports flag rate per threshold. Refuses on extractor mismatch.
- `threshold_sweep_full.py`: Full sweep 0.05-0.95 with tp/fp/fn to CSV.
- `eval_grouped.py`: Diagnostic, not shipped metrics. Random split vs grouped (whole domains held out) vs leave-one-domain-out, plus a coefficient audit. `--exclude` drops a batch before training, so running with and without it measures what that batch changed under identical code.

**`validation/`**
- `sample_deployment_validation.py`: Scores a pool, draws a uniform random sample of what clears the threshold for hand-labeling. Random rather than confidence-sorted, since the top of the ranking would flatter the estimate. Loads a saved model rather than fitting one, so a reported number always names the model that produced it.
- `scan_domain_breakdown.py`: Groups flagged pages by registered domain. A model can hold good precision while firing on only two or three sites. Also loads a saved model rather than fitting one.
- `precision_report.py`: The shipped metric. Joins the hand labels to their scores, weights each label by its stratum's frame size, and reports precision, contamination, recall and yield by threshold with bootstrap intervals, then projects them onto every eligible page in the crawl. Reads scores already produced by `score.py`; does not load a model itself.

**`archive/rule-based/`** — file inventory for the superseded classifier. Why it was dropped, and why it still runs as the `original` batch's prefilter, is under [Approaches that were dropped](#approaches-that-were-dropped).
- `URL_Classifier.py` (whitelist then hostname keyword match), `WL_Builder.py` (discovers candidate domains via Athena for manual triage), `wl_candidates.txt`, `CC_Classifier_Test.py` (samples and classifies for manual review), `cl_validation_results.txt` (recall validation against CourtListener bulk data).

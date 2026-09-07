# cc-legal-audit

Empirical audit of how uniform MinHash fuzzy deduplication thresholds affects semantic coverage (topic entropy) in the legal domain versus general web text, using Common Crawl data.

## Motivation

Standard LLM pre-training pipelines apply a uniform Jaccard similarity threshold (typically 0.8 on 13-grams) inherited from Gopher without empirical validation across domains. Legal text has constrained vocabulary and structural conventions that inflate n-gram similarity scores. Substantively different documents get flagged as duplicates not because they share content, but because they sound alike. DCLM noted this as an open problem and left domain-level investigation as future work. This project picks that up.

## Methodology

1. Sample a Common Crawl snapshot (CC-MAIN-2026-12)
2. Apply fixed preprocessing: language filtering, quality heuristics, repetition removal (Gopher defaults held constant)
3. Classify documents into legal and general web subsets via URL tokenization
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

The result that established the approach, recorded 2026-09-03 on 4044 labels. Current numbers are under "Bills batch" and "Publisher weighting" below.

Same labels, same evaluation script, same splits as the URL model. The only thing that changes is what the model reads: the extracted page text from each URL's Common Crawl capture, instead of the URL string.

**Why.** The URL classifier could not recognize a legal publisher it had no training examples from. Leave-one-domain-out recall was 0.000 at every usable threshold. Page text has cross-publisher signal that URL strings do not: statutory prose from Kansas reads like statutory prose from Florida, while `ksrevisor.gov` and `flsenate.gov` share nothing as strings.

**Pipeline.**

1. `src/corpus/fetch_warc_text.py` joins the labeled URLs to the Common Crawl index through Athena to get each capture's WARC filename, byte offset and length, then range-fetches those bytes over HTTP and extracts text with trafilatura. 3996 of 4044 labeled URLs (98.8%) are present in CC-MAIN-2026-12, and 3787 (93.6%) produced usable text: 902 legal / 2885 non_legal. Per-publisher coverage is 92 to 100%, so no domain is too depleted to hold out fairly.
2. `src/classifier/features.py` holds both feature recipes in one place so training, scoring and evaluation cannot drift onto different settings. `url` is the existing char 3-5 gram vectorizer, unchanged. `text` is word unigrams and bigrams over the page body.
3. `train_classifier.py --features url|text` reports held-out metrics, then refits on all labels and saves the fitted vectorizer and model to `models/<mode>_clf.joblib`. The bundle records the label file's hash, the trafilatura version that produced the training text, and the library versions, so a score can be traced to the model that produced it.
4. `score.py` scores any jsonl with a saved bundle. This is what lets the model be pointed at unlabeled crawl pages rather than only at its own test set.
5. `eval_grouped.py --features url|text` runs identical splits and metrics against either, which is what makes the two comparable.

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
- The positive class is register-narrow. Tagging all 955 legal labels by the labeling agent's own rationale: 660 statute or code (69%), 433 regulation (45%, overlapping), against 37 bills (3.9%), 28 court opinions (2.9%) and 3 filings. The 28 opinions sit almost entirely in cornell.edu (16) and judiciary.uk (10), so no publisher in the set has opinions as its dominant register. The model now generalizes across publishers, but the corpus it was trained on is mostly statutes and regulations.
- Precision has not been measured on the real crawl distribution. Every number above comes from a label set that is about 23% legal; the crawl is far below that. Precision does not transfer across that gap, only recall and false-positive rate do, so the 0.90+ figures should not be read as deployment precision. The measurement to trust is a hand-labeled random draw from what the model flags on a uniform crawl sample. A first pass at this is recorded under "Flag rate on a crawl sample" below; the full labeled version is underway.

### Bills batch (recorded 2026-09-07)

400 URLs from state legislature bill sections, 16 from each of 25 states. The agent returned 375, skipped 25, and labeled **only 32 legal**. 350 of the 400 came from inside a bill section, so roughly 9% of pages filed under a legislature's bill path are bill text; the rest are status pages and indexes that list a bill's sponsors, actions and votes and link to the text elsewhere, which is NON_LEGAL because the page's own HTML carries no legal text.

Label set after merge: 4419 URLs, 4338 with usable text, 987 legal / 3351 non_legal across 36 legal-bearing domains, 21 with the 5 legal rows leave-one-domain-out needs (up from 16). Top-3 publisher share of the legal class fell 79.3% to 76.7%.

**Effect**, measured on the 14 publishers whose legal-row count did not change, so held-out rows are identical and only training differs: macro LODO recall (t=0.85) **0.419 to 0.436**. Six improved, eight held, none got worse, and every gain is on a publisher the batch did not touch (cornell.edu 0.163 to 0.195, virginia.gov 0.402 to 0.433, ohio.gov 0.200 to 0.267), so this is generalization rather than the model learning the new rows. The all-publisher macro moved 0.421 to 0.399, but that averages a different set: five new domains entered and idaho.gov (6 legal rows) scores 0.000 at every setting.

32 legal labels for 400 URLs is a poor yield. The effect is real, consistent, and small.

### Publisher weighting

Three publishers hold 77% of the legal class, so most of what the fit learns comes from three sites. `domain_weights()` in `features.py` weights each row `(1 / rows its publisher contributed to its class) ** alpha`, rescaled to average 1 so regularization strength does not move with alpha. At alpha 0, justice.gc.ca's 342 legal rows outvote nmlegis.gov's 5 by 68 to 1; at 0.5 by 8 to 1; at 1.0 not at all. This is group imbalance, not leakage: leave-one-domain-out already holds each publisher out entirely.

| alpha | macro LODO recall (21 publishers, t=0.85) |
|---|---|
| 0 | 0.399 |
| **0.5 (default)** | **0.475** |
| 1.0 | 0.438 |

At 0.5, 14 publishers improved, 7 held, none got worse, including all three large ones. 1.0 overcorrects, letting a 5-row publisher carry as much weight as a 342-row one and follow noise. Grouped-split recall rises at every threshold (0.50: 0.711 to 0.741; 0.65: 0.533 to 0.567; 0.85: 0.207 to 0.219) with precision flat.

The coefficient audit shows the mechanism: Justice Canada furniture is demoted and register vocabulary replaces it. `marginal` 6th to 25th, `modified` 8th to 20th, `note` and `details` out of the top 30; `section` 10th to 3rd, `chapter` 21st to 9th, with `statutes` and `amended` entering.

**Caveat for any write-up.** Alpha 0.5 was chosen by reading macro LODO recall and macro LODO recall is what then gets reported, so 0.475 is optimistically biased for unseen publishers. That is the winner's curse, not leakage. A nested selection (choosing alpha inside each fold from training publishers only) would remove it and has not been run.

**Held-out random split at alpha 0.5** (868 rows, 197 legal / 671 non_legal). Publishers sit on both sides, so read as a ceiling:

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.50 | 0.691 | 0.944 | 0.798 |
| 0.60 | 0.769 | 0.893 | 0.826 |
| 0.65 | 0.835 | 0.848 | 0.841 |
| 0.75 | 0.925 | 0.685 | 0.787 |
| 0.85 | 0.979 | 0.477 | 0.642 |

### Flag rate on a crawl sample

5000 URLs drawn uniformly from `raw_pool.jsonl`, text fetched through the training extractor, scored with the saved model. 4662 produced usable text.

| threshold | flagged | flag rate | distinct domains |
|---|---|---|---|
| 0.50 | 29 | 0.622% | 22 |
| 0.60 | 18 | 0.386% | 15 |
| 0.75 | 10 | 0.215% | |
| 0.85 | 6 | 0.129% | 5 |

**The concentration test passes.** 18 flagged pages across 15 publishers, against the URL model's 24 flagged pages that were all justice.gc.ca. Hits include virginia.gov, cornell.edu, justice.gc.ca, gazette.gc.ca, wi.gov, ecfr.io, parliament.uk and tcilii.org. It also flagged a Privy Council judgment off 28 opinion labels total, the only evidence so far on whether the missing opinion register is fatal.

This section's precision estimates were superseded by the measurement below, which replaced an eyeball of 18 URLs with 152 labels.

**The dominant false positive is one nameable class:** legal-sounding text written by a private party. A purchase order's terms and conditions on `lifesafetyservices.com` scored 0.957, second-highest of all 4662 pages. The rest are terms of service, privacy policies, institutional codes of conduct and commercial summaries of state law. Behind that: government pages that are not law (agency tool catalogs, facility inspection records) and Cornell LII definition popups, the family already corrected twice in `intake.py`. The definition excludes all of these on its first half, which asks who published the page rather than how it reads, but no training negative teaches that, because every earlier batch sampled from publishers of law.

**Caveat.** `raw_pool.jsonl` predates the sampler fix noted under `fetch_candidate_urls.py`, so it is not a clean uniform draw. Directional, not publication-grade.

### Deployment precision, measured (2026-09-07)

32,000 URLs drawn uniformly from the pool, every page fetched and scored, and
every flagged page kept and hand-labeled by the labeling agent. 169 candidates,
152 labeled, 17 skipped: **27 legal, 125 non_legal.**

| threshold | flagged | legal | precision | 95% CI |
|---|---|---|---|---|
| 0.50 | 152 | 27 | 0.178 | [0.125, 0.246] |
| **0.60** | 93 | 25 | **0.269** | [0.189, 0.367] |
| 0.75 | 42 | 17 | 0.405 | [0.270, 0.555] |
| 0.85 | 18 | 11 | 0.611 | [0.386, 0.797] |
| 0.90 | 12 | 8 | 0.667 | [0.391, 0.862] |

Precision rises monotonically with score, so the model **ranks** correctly. What
is wrong is where the decision boundary falls. No threshold is usable: 0.90
reaches only 0.667 and flags 12 pages in 32,000, which scales to roughly 220
documents from the whole pool.

**Base rate: 0.12%.** 27 legal found, grouped-split recall 0.741 at t=0.50, so
about 36 legal pages among the ~29,800 with text. The earlier 0.3% estimate came
from eyeballing 18 URLs and was wrong; the original 0.1% assumption was right.

**Why no amount of hard-negative mining closes this.** Solving
`precision = (base x recall) / (base x recall + (1-base) x FPR)` at base 0.12%
and recall 0.5 gives the false-positive rate each target needs:

| target precision | required FPR | vs. measured 0.227% |
|---|---|---|
| 80% | 0.015% | 15x reduction |
| 90% | 0.0067% | 34x reduction |

**The false positives are one class:** legal-sounding text written by a private
party. Top scorers were an SEC filing exhibit (a contract) at 0.976, a tax-treaty
commentary site at 0.960, a purchase-order terms page at 0.957, a news article
about a city ordinance at 0.923, and a law firm article at 0.871. Behind those:
terms of service, privacy policies, institutional codes of conduct, auction
catalogue terms, and Cornell LII definition popups. The LEGAL definition excludes
all of them on its first half, which asks who published the page rather than how
it reads, but no training negative taught that, because every earlier batch
sampled from publishers of law.

### Retraining on the mined negatives

The 152 labels went back into training (one retrain, by design). Label set:
4,490 rows, 1,014 legal / 3,476 non_legal, 42 legal-bearing domains.

Measured with `--exclude` so both sides run identical code, on the 21 publishers
present in both, at `--domain-weight 0.0`:

| | macro LODO recall |
|---|---|
| without flagged batch | 0.399 |
| with flagged batch | 0.375 |

Eight publishers fell, thirteen held, none rose. The headline macro reads
0.399 to 0.398 only because `wisconsin.gov` entered the table at 0.889 and masked
the decline. Grouped-split precision improved at the low thresholds (0.50:
0.457 to 0.492; 0.65: 0.529 to 0.567).

That is what hard negatives do: the model learned that legal-sounding text is
often not law and became more cautious, buying precision with recall. The
question was only ever whether it bought enough.

**It did not.** Rescoring the same 152 pages with the retrained model gives 0.711
precision at t=0.60, but those pages are now in its training set, so that is a
memorisation ceiling rather than a measurement, and real precision on unseen
pages is strictly below it. Even with them in training the model still scores the
tax-treaty site at 0.854, the SEC contract exhibit at 0.829 and the purchase-order
terms at 0.715, which suggests the class is not linearly separable from primary
law in this feature space.

### Decision: source the legal bucket by publisher, classify page type within it

Open-web detection fails for one reason, and it is not the model. At a 0.12% base
rate, Bayes requires a false-positive rate near 1e-4 before precision can reach
0.9. Restricting the candidate pool to legal-publisher domains removes that
problem instead of fighting it.

| | open-web pool | authority-domain pool |
|---|---|---|
| base rate | 0.12% | 23.3% |
| measured precision @0.60 | 0.269 | see below |

**The authority-domain pool has a 23.3% base rate, which is the label set's own
base rate.** That is what makes the held-out random-split numbers valid there
with no base-rate translation, which is the step that destroyed every projection
on the open-web pool:

| threshold | precision | recall |
|---|---|---|
| 0.65 | 0.828 | 0.783 |
| 0.75 | 0.937 | 0.586 |
| 0.85 | 0.964 | 0.394 |

Of a uniform 32,000-page crawl draw, only 119 pages (0.37%) sit on an authority
domain. The filter discards 99.6% of the crawl and raises the base rate roughly
200-fold.

**This is not the domain whitelisting this project rejected.** That objection was
to a whitelist as a *classifier*, because a hostname cannot tell a statute from a
homepage or a docket index, and it still stands. Using enumerated publishers to
*source* a corpus is a different decision, and the classifier still does the work
the host list cannot: only about 9% of pages captured inside a legislature's own
bill section are bill text. Sourcing narrows the haystack; the classifier finds
the needle.

**Both buckets remain Common Crawl.** The authority supplies hostnames only,
never URLs or documents. Every page is a CC capture through the same WARC fetch,
the same pinned trafilatura settings and the same Gopher filters, exactly as
`fetch_cl_urls.py` and `fetch_bill_urls.py` already work. The general-web bucket
should be drawn by the same host-level procedure over random hosts, so both sides
of the comparison are defined the same way.

**What this costs, and what has to be said in the write-up.** The corpus becomes
"legal text from known legal publishers," not "all legal text in Common Crawl."
Three claims are given up: what fraction of the crawl is legal, representativeness
across the whole web, and anything about legal text hosted off publisher domains.
Corpus composition also stops being defensible by appeal to uniform sampling and
becomes a choice: 69% statute, 45% regulation, 3.9% bills, 2.9% opinions. The
unsourced opinion register matters more under this plan than it did before, and
entropy should be reported per register so a low figure cannot be dismissed as a
composition artifact.

**The classifier work is not discarded.** It produced a measured negative result:
open-web legal document detection at a 0.12% base rate needs a false-positive
rate near 1e-4, and TF-IDF plus logistic regression on 1,014 positives reaches
0.227%, giving 27% precision at the operating threshold. That result is what
justifies the sourcing decision.

### Sourcing the missing registers

The register gap above is being closed by asking an outside authority where each publisher files its documents, rather than picking URL patterns by hand. The authority supplies the shape (which site, which section of it); Common Crawl supplies the actual candidate URLs, so nothing is sampled from the authority itself and there is no distribution shift. `src/samples/fetch_register_sources.py` does that step for both registers.

**Bills: works.** Open States covers all 50 state legislatures and records a `sources` link back to each bill's page on its own state site. All 50 states are queried, so which ones are worth using gets decided later from measured crawl depth rather than guessed at up front.

**Court opinions: this route does not work.** Measured over 220 CourtListener opinion records, 96% carry no `download_url` at all. Its corpus is overwhelmingly bulk donations (the schema carries `html_columbia`, `xml_harvard`, `html_lawbox`), so CourtListener holds the opinion text but has no record of a page it came from. The 9 records that did carry an address gave a govinfo PDF directory, an upload folder and PACER's paywalled gateway, none of them a court's opinion section. That is a property of the source rather than the sample size, so the opinion side needs a different authority or a different method.

A PDF address still counts as a signpost, since it says which section a publisher files under and that section usually holds HTML too. Collecting PDFs *as documents* is a separate unsettled decision: it needs a second extractor, and PDF-to-text differs from HTML-to-text in hyphenation, repeated page furniture and whitespace, which is the surface variation 13-gram MinHash responds to. Doing that on the legal side only would confound the study's headline comparison.

## Currently underway

**Defining the authority domain list.** `fetch_cl_hostnames.py` currently holds
110 hostnames across 61 registered domains: 60 courts sampled from the
CourtListener courts API plus a closed enumeration of all 50 state legislature
sites. The 60 is a *strided sample*, taken because the API's hourly cap makes a
full pull slow. For the pivot the list stops being a sampling frame and becomes
the corpus definition, so it needs the complete pull (`SAMPLE_PAGES = None`).
`register_sources_bills.jsonl` adds 49 legislature sites with the specific path
sections where bills live, which is a tighter second filter.

**Open questions.**

- The general-web bucket should be redrawn by host to match how the legal bucket
  is defined, rather than by uniform page sampling.
- Opinions remain unsourced (see above), and matter more under this plan.
- `OPERATING_THRESHOLD` in `eval_grouped.py` is hardcoded to 0.85, the URL
  model's operating point, so every leave-one-domain-out figure above is measured
  at a threshold the text model does not run at. Before/after comparisons are
  unaffected, both sides use the same constant.
- Whether to include PDFs in either bucket, which needs Common Crawl payload
  truncation measured first.
- Precision inside authority domains has not been measured on held-out data. The
  random-split figures above transfer on a base-rate argument, but a labeled draw
  from the domain-filtered pool would measure it directly and is far cheaper than
  the open-web equivalent, since the base rate is 200x higher.

## Code

```
src/
  samples/     draw URL samples from Common Crawl, build labeling batches
  labels/      merge raw labeling-agent output into clean label files
  corpus/      fetch and extract page text from Common Crawl
  classifier/  train, score and evaluate the TF-IDF/LR model
  validation/  deployment-distribution precision sampling
```

Each folder is named for what it produces: samples go out to the labeling agents, labels come back.

**`src/samples/`**
- `fetch_candidate_urls.py`: Draws a uniform random URL sample from a CC snapshot, writes `data/candidates/raw_pool.jsonl`. Sizes the population with a count first, then samples with `TABLESAMPLE BERNOULLI(p)` and no `LIMIT`: a `LIMIT` returns whichever index files answered first, which is not a uniform sample and cannot be repaired by shuffling afterwards. Each row carries its WARC pointer, so page text can be fetched without a second index lookup.
- `build_label_batch.py`: Turns that raw pool into the batch handed to the labeling agent: mixes rule-based prefilter hits (to boost legal density, since legal pages are ~0.2 to 0.3% of the raw crawl), raw random URLs, and synthetic homepage URLs for whitelisted domains (deliberate hard negatives for the index-vs-filing failure mode). Writes `data/candidates/candidates.jsonl`.
- `fetch_targeted_urls.py`: Error-driven follow-up pull, not random, it targets the specific domains/URL-shapes the false-negative analysis flagged. Writes `data/candidates/targeted_batch.jsonl`.
- `fetch_raw_pool_no_ca.py`: Second Athena pull excluding `.ca` domains, used to source the second deployment-validation sample from a different slice of the crawl.
- `fetch_cl_hostnames.py`: Builds a directory of likely-legal hostnames from the CourtListener courts API plus a closed enumeration of all 50 state legislature sites. Writes `data/candidates/court_hostnames.jsonl`.
- `count_cl_captures.py`: Ranks those hostnames by actual Common Crawl capture depth, since CourtListener docket size and crawl coverage are uncorrelated. Writes `data/candidates/cc_host_counts.jsonl`.
- `fetch_cl_urls.py`: Draws a per-publisher URL sample from Common Crawl for the hostnames above (random URLs for hard negatives, one URL per path prefix for section coverage). Writes `data/candidates/host_sample_batch.jsonl`.
- `fetch_register_sources.py`: Asks Open States, for each of the 50 states, where that legislature files its bills, then reduces each bill's address to a site plus a wildcard path pattern (`/li/%/measures/%`) and counts how many records back each one. Reports what the authority says and touches neither Common Crawl nor the documents themselves. Writes `data/candidates/register_sources_bills.jsonl`.
- `fetch_bill_urls.py`: Turns those sections into a labeling batch. One Athena query sorts every captured page on those sites into `in` or `out` by whether its path matches the site's pattern, drops sites with under 200 captured pages inside, then takes a fixed quota from each so a deeply-crawled state cannot dominate. Writes `data/candidates/bill_sample_batch.jsonl`.
- `fetch_flagged_urls.py`: Draws a uniform random sample of the crawl, fetches and scores every page, and keeps the ones the model flags. Unlike every other sampler here it does not sample where law is expected to be, which is what makes its negatives the only ones drawn from the distribution the model actually fails on. Writes the URLs and the scores to separate files so the labeling agent cannot see the model's confidence.
- `athena.py`: Shared Athena query-runner that also reports bytes scanned and estimated cost.

**`src/labels/`**
- `intake.py`: Merges all five labeling passes' raw per-worker output into one file, `data/processed/labeled_urls.jsonl`, tagging each row with a `source` field (`original` / `target` / `cl` / `bills` / `flagged`). Applies the documented label corrections (link-only pages, index/landing pages, definition popups) per pass.

**`src/corpus/`**
- `fetch_warc_text.py`: Resolves each URL to its Common Crawl capture (Athena, cached to `data/processed/warc_pointers.jsonl`), range-fetches the WARC record over HTTP, and extracts text with trafilatura. Takes `--input/--output`, so the label set and any unlabeled crawl sample go through identical extraction. Every input URL gets an output row, carrying a `skip_reason` when extraction failed, so counts always reconcile against the input.

**`src/classifier/`**
- `features.py`: The two feature recipes, `url` (char 3-5 grams on the URL string) and `text` (word 1-2 grams on the page body), plus the domain purity filter, the publisher weighting and label loading. Kept in one file so training, scoring and evaluation cannot drift onto different settings.
- `train_classifier.py`: Trains on the merged label file with `--features url|text` and `--domain-weight` (default 0.5), reports held-out metrics and a threshold sweep, then refits on all labels and writes the fitted vectorizer and model to `models/<mode>_clf.joblib` with provenance (label file hash, extractor version, weighting, library versions).
- `score.py`: Scores any jsonl with a saved bundle and reports the flag rate per threshold. Refuses to run when the input text came from a different extractor than the model was trained on.
- `threshold_sweep_full.py`: Full precision/recall/F1 sweep across every threshold from 0.05 to 0.95, with raw tp/fp/fn counts, written to `data/processed/threshold_sweep_<mode>.csv` for the record.
- `eval_grouped.py`: Diagnostic, not the shipped metrics. Takes `--features url|text` so both models are scored by identical code, plus `--domain-weight` and `--exclude` (a jsonl of URLs to drop before training, so running with and without a batch measures what that batch changed using the same code for both numbers). Compares a random row split against a grouped (whole-domain-held-out) split and a leave-one-domain-out sweep, plus a coefficient audit, to check whether the classifier generalizes to unseen legal publishers or just recognizes known ones.

**`src/validation/`**
- `sample_deployment_validation.py`: Scores an unlabeled pool with a saved model, then draws a uniform random sample of what clears the threshold for hand-labeling. Random rather than confidence-sorted, since the top of the ranking is the easy part and would flatter the estimate. Writes the URLs and the scores to separate files so a probability cannot anchor the manual judgment, and prints the flag rate, which bounds precision before any labeling happens. Takes `--model/--pool`, replacing the earlier pair of near-identical per-pool scripts.
- `scan_domain_breakdown.py`: Scans a pool at a given threshold and groups the flagged pages by registered domain, to see which publishers the model's confident predictions concentrate on. A model can hold good precision while only ever firing on two or three sites, which would make the study's legal bucket those sites rather than legal text.

Both load a saved model rather than fitting one, so a reported number always names the model that produced it.

### Archive: rule-based classifier (`archive/rule-based/`)

Superseded by the char n-gram TF-IDF + logistic regression approach (see URL Classifier above), kept for reference.

- `URL_Classifier.py`: URL-based legal/non-legal classifier. Two-layer architecture: curated domain whitelist (`wl_candidates.txt`) checked first, then strict keyword matching on the hostname only (path ignored to prevent false positives).
- `WL_Builder.py`: Discovers candidate legal domains from a CC snapshot via Athena. Queries `url_host_name` grouped by page count and writes results to `wl_candidates.txt` for manual triage.
- `wl_candidates.txt`: Triaged whitelist of primary legal source domains (courts, legislatures, statute repositories). One entry per line; suffix matching at runtime covers all subdomains.
- `CC_Classifier_Test.py`: Samples URLs from a CC snapshot via Athena TABLESAMPLE, classifies them, and prints positives tagged `[WL]` or `[KW]` plus a negative sample for manual precision/recall review.
- `cl_validation_results.txt`: External recall validation of the rule-based classifier against CourtListener bulk opinion data.

## Paper
University of Florida undergraduate research.

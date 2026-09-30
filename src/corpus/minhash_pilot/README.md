# MinHash pilot

A small, hand-inspected check run before the full crawl, for the audit described in the [top-level README](../../../README.md).

**Result (2026-09-29): go, with preliminary evidence of asymmetric deduplication.** Among near-duplicate pairs that a 0.7 threshold would remove, 50% of legal pairs are distinct content (different documents, or different versions of one) against 19% of control pairs: a gap of **+31 points** (95% CI +14 to +47). The pre-set go rule needed +15 with the interval above 0. Most of the legal gap in this sample comes from one publisher; see [Caveats](#caveats).

## The concern

> Before the full crawl run, please do a small pilot and manually inspect some MinHash duplicate pairs. We need to see whether legal pairs are truly duplicates or just different documents with similar structure/boilerplate.

**Why it matters.** The study's premise is that a uniform MinHash threshold (Jaccard 0.8 on word 13-grams, inherited from Gopher) collapses distinct legal documents more often than general web pages, because legal text shares vocabulary, templates and boilerplate. If the pairs MinHash flags in legal text are real copies, there is no asymmetry to measure, and the full run (topic fits, a threshold sweep, three dedup pools) measures nothing.

**What this is, and isn't.** A check that the mechanism exists before paying for the full run. The numbers come from 110 inspected pairs, after Gopher filtering but before the rest of the pipeline, so they are not estimates of what the full run will find.

## Method (run 2026-09-28)

**Environment.** Project venv for everything except step 3, which runs in `laya-env` ([setup in the classifier README](../../classifier/README.md#local-judge-laya-fine-tuned-on-jev-frozen-2026-09-25-cut-085)). Both models are gitignored and must be copied over: `models/laya-legal` (the judge) and `models/text_clf.joblib` (the screen). Steps 1 and 4 query Athena, so `.env` needs AWS credentials and `ATHENA_OUTPUT_LOCATION`. All commands run from the repo root; every random draw is seeded.

**1. Draw candidate legal URLs.** `python src/corpus/minhash_pilot/fetch_legal_pages.py --total 5000` → `legal_pool.jsonl`. Random pages from the 19 sites where Laya found legal text on the open web, each site getting its share of the 5,000 in proportion to how much legal text Laya found there. Drawn at 3× (14,954 URLs), since not every page on a legal site is legal.

**2. Fetch their text.** `python src/common/fetch_warc_text.py --input data/minhash_pilot/legal_pool.jsonl --output data/minhash_pilot/legal_pool_text.jsonl`.

**3. Classify.** `C:\projects\laya-env\Scripts\python.exe src/classifier/largeModel/score_laya.py --model models/laya-legal --max-len 1024 --input data/minhash_pilot/legal_pool_text.jsonl --output data/minhash_pilot/laya_legal_scores.jsonl` (add `--batch-size 2` on a small GPU). The frozen cascade: the TF-IDF screen at 0.30 (on by default) passed 13,137 pages to Laya.

**4. Build both sets.** `python src/corpus/minhash_pilot/fetch_control_pool.py`.
- *Legal:* pages Laya kept (p ≥ 0.85), trimmed back to each site's target: **4,800 pages**. Eight sites fell short of their target, the small ones most sharply (`townhall.virginia.gov` kept 1 of 29), because most of their pages are not legal.
- *Control:* for each legal site, one random non-legal site of about the same size (pages in the crawl, within a factor of two), with the same number of pages drawn from it: **4,787 pages** from 19 sites. Matched by site because near-duplicates mostly come from a site reusing its own templates; a random draw across the web would contain almost none.

**5. Fetch control text.** `python src/common/fetch_warc_text.py --input data/minhash_pilot/control_pool.jsonl --output data/minhash_pilot/control_text.jsonl`.

**6. Filter, then find pairs.** `python src/corpus/minhash_pilot/minhash_pairs.py` → `pairs.jsonl`.
- *Gopher first.* Both sets go through Gopher's 8 quality and 13 repetition rules ([`../filters/gopher.py`](../filters/gopher.py), the paper's thresholds), since a real pipeline only deduplicates pages that pass them.
- *Pairs.* Each page becomes the set of its 13-word phrases; two pages' similarity is the share of phrases they have in common (Jaccard). MinHash (128 permutations) screens out obvious non-matches, exact similarity is computed for the rest, and every pair at ≥ 0.6 within a set is kept.

| set | pages | pass Gopher | main reasons for dropping |
|---|---|---|---|
| legal | 4,800 | 3,136 (65%) | duplicate 5-grams 1,071; alphabetic words 240; bullet lines 116 |
| control | 4,783 | 2,188 (46%) | word count 939; alphabetic words 596; hash ratio 337; stop words 275 |

| pairs found | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9+ | same site |
|---|---|---|---|---|---|
| legal | 45 | 20 | 15 | 47 | 98% |
| control | 463 | 180 | 67 | 4,533 | 100% |

**7. Draw the labeling batch.** `python src/corpus/minhash_pilot/build_pair_batch.py` → `pair_batch.jsonl` and `pair_scores.jsonl`. Up to 30 pairs per set per similarity band, from 0.7 up; no site pair supplies more than 10 in a band, so one site's template can't fill the batch; no page appears twice in a band. **110 pairs.** Set and similarity go to a separate file, so labeling is blind to both. The 41 pairs whose text is identical word for word are labeled `same` automatically.

**8. Label.** `python src/corpus/minhash_pilot/label_pairs.py` shows each pair as a word diff (only what differs, with context).

| label | meaning |
|---|---|
| `same` | Operative text identical; only identifiers or metadata differ (section, bill or case numbers, agency or court name, authority citations, history notes, dates, URLs, punctuation, site chrome) |
| `version` | The same instrument at different times: amended, or introduced vs. enrolled |
| `different` | Operative text differs beyond identifiers: a page has provisions, holdings or other substance the other lacks, even if most of the page is shared template |
| `exclude` | Can't judge: extraction broke, or a page is chrome only |

*Tie-breakers.* One unique operative sentence is enough for `different`; shared chrome alone never is. A different name or number on the same text is not new content. When one page is contained in the other (a page of an act vs. its full text, same date), the pair is `same`. The full rules print in the labeler (`h`).

**9. Report.** `python src/corpus/minhash_pilot/pilot_report.py`.

**Go rule** (set before the report was run). At similarity ≥ 0.7 pooled (0.7 is the lowest threshold the full run sweeps, and pooling from it up covers the 0.8 and 0.9 bands too, including the usual 0.8 threshold; 0.9+ is also reported on its own), compare the share of `different` + `version` pairs (excluded pairs left out; auto-labeled pairs counted):
- *Go:* legal − control ≥ 15 points, with the 95% interval above 0.
- *Stop:* the gap is under 15 and the interval's upper end is under 15.
- *Inconclusive:* anything else. Label one more batch (`--per-band 50 --site-cap 30`) and apply the same rule to the pooled labels, once.

## Results (2026-09-29)

| set | band | pairs | same | version | different | exclude | not same | 95% CI |
|---|---|---|---|---|---|---|---|---|
| legal | 0.7 | 15 | 1 | 3 | 11 | 0 | 93% | [0.70, 0.99] |
| legal | 0.8 | 10 | 3 | 1 | 6 | 0 | 70% | [0.40, 0.89] |
| legal | 0.9 | 29 | 23 | 0 | 6 | 0 | 21% | [0.10, 0.38] |
| control | 0.7 | 15 | 7 | 0 | 7 | 1 | 50% | [0.27, 0.73] |
| control | 0.8 | 11 | 8 | 0 | 3 | 0 | 27% | [0.10, 0.57] |
| control | 0.9 | 30 | 29 | 0 | 0 | 1 | 0% | [0.00, 0.12] |

| main test, ≥ 0.7 pooled | legal | control | gap | 95% CI |
|---|---|---|---|---|
| **different + version** | **27/54 = 50%** | **10/54 = 19%** | **+31** | **[+14, +47]** |
| different only | 23/54 = 43% | 10/54 = 19% | +24 | [+7, +40] |

**How to read the tables.** "Not same" means the two pages carry different content (labeled `different` or `version`), so deduplicating one of them would throw real content away. The gap is legal's not-same share minus control's.

**What it means: preliminary evidence of asymmetric deduplication.**
- **Legal near-duplicates are more often distinct content.** At similarity 0.7 and up, half of legal pairs (50%) are distinct content under the labeling rules, against about one in five control pairs (19%). A 0.7 threshold would merge distinct legal pages about 2.5 times as often.
- **It isn't only amended versions.** Counting only pairs that are entirely different documents, legal is still 43% against 19% (+24).
- **It shows up even at high similarity.** At 0.9 and up, above the usual 0.8 threshold, 6 of 29 legal pairs are different documents; none of 30 control pairs are.
- **So, to the mentor's question:** in this sample, many legal pairs that MinHash calls duplicates are different documents with shared structure, and this happens much more in legal text than in the control.

**Go.** The rule needed a gap of at least +15 with the interval above 0. The gap is +31, interval +14 to +47, so the full run goes ahead.

### Caveats

Each caveat limits what the result above can claim, and says what the full run should do about it.

**1. Most of the legal gap comes from one site.**
- *What:* `docs.legis.wisconsin.gov` supplies 22 of the 54 legal pairs and 19 of the 27 not-same ones. Its pages show each section inside the text of the surrounding chapter, so neighbouring sections overlap heavily and look like near-duplicates.
- *Why it matters:* without that site, the gap drops to +6 (interval −11 to +25), which is not distinguishable from zero. So this sample shows the effect clearly for one site's page layout, not for legal text in general. It also can't say how that site as a whole, or other sites built the same way, behave: 22 pairs is too few.
- *Full run:* report results per publisher, not just pooled.

**2. The sample is small.**
- *What:* 54 judged pairs per set, and 10 to 30 per band.
- *Why it matters:* every interval is wide (about ±17 points on the gap). The go decision is clear, but the exact size of the gap isn't.

**3. Gopher removes different pages from each set.**
- *What:* before any comparison, 35% of legal pages failed Gopher, mostly for repeating 5-word phrases (22% of all legal pages). Control pages failed for other reasons: too short, too few letters, too many hashes, too few stop words.
- *Why it matters:* MinHash only sees what Gopher keeps, and Gopher treats the two domains differently.
- *Full run:* report Gopher's drops per domain next to MinHash's.

**4. Which Gopher implementation is used changes a lot.**
- *What:* [`../filters/gopher.py`](../filters/gopher.py) follows the paper's rules. datatrove, the library FineWeb uses, counts punctuation marks as words, so citation-heavy legal pages fail its "80% of words contain a letter" rule. On this sample it keeps 28% of legal pages and 30% of control, against 65% and 46% here.
- *Why it matters:* "Gopher filtering" is not one fixed thing; results depend on whose version runs.

**5. This is not the full pipeline.**
- *What:* pages here went through Gopher only, and the control is small and site-matched.
- *Why it matters:* other preprocessing, and the full run's much larger general bucket, can change which pairs MinHash sees. These numbers are a check that the effect exists, not a prediction of its size.

**6. The sample covers a narrow slice.**
- *What:* legal pages come from 19 sites, so rarer legal publishers are missing, and 98% of legal pairs are within one site, so copies of the same law on different sites are barely tested. The control counts as "non-legal" only because its sites aren't on `data/candidates/legal_domains.jsonl`; its pages were never classified.
- *Why it matters:* cross-site legal duplicates, the case a crawl-wide dedup hits most, are largely untested here.

## Code

- `sampling.py`: Shared settings and helpers every script imports: the snapshot and seed, what counts as a site (hostname without `www.`) and an eligible page (English HTML, fetched OK), and the Athena query that draws random pages per site.
- `fetch_legal_pages.py`: Step 1. Candidate legal URLs from the 19 sites, in proportion to Laya's open-web keeps.
- `fetch_control_pool.py`: Step 4. The final legal set, and the site-matched control URLs.
- `minhash_pairs.py`: Step 6. Gopher filtering via `../filters/gopher.py`, then every near-duplicate pair at ≥ 0.6 within each set. Also defines the similarity bands the next two scripts use.
- `build_pair_batch.py`: Step 7. The blind labeling batch, with word diffs and auto-labels for identical text. A rebuild keeps labels for pairs still in the batch.
- `label_pairs.py`: Step 8. Terminal labeler; holds the labeling rules.
- `pilot_report.py`: Step 9. Label counts per band and the legal − control gap. Console only.

Steps 2, 3 and 5 use `../../common/fetch_warc_text.py` and `../../classifier/largeModel/score_laya.py`.

**Data** (`data/minhash_pilot/`). The three files with page text are gitignored; they regenerate from the committed pointer files with no further Athena spend.

| file | holds | committed |
|---|---|---|
| `legal_pool.jsonl` | candidate legal URLs, each site's target, where to fetch each page | yes |
| `legal_pool_text.jsonl` | their extracted text | no |
| `laya_legal_scores.jsonl` | Laya's score for each page that passed the screen | yes |
| `legal_set.jsonl` | the final legal pages, with text | no |
| `control_pool.jsonl` | control URLs and where to fetch them | yes |
| `control_sites.jsonl` | which control site stands in for which legal site | yes |
| `control_text.jsonl` | control page text | no |
| `pairs.jsonl` | every pair at ≥ 0.6: set, URLs, sites, similarity | yes |
| `pair_batch.jsonl` | the 110 labeled pairs with their diffs | yes |
| `pair_scores.jsonl` | each batch pair's hidden set and similarity | yes |

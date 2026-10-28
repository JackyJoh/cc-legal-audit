# corpus

The full run: pulls pages from Common Crawl, filters and labels them, removes near-duplicates at each threshold, and measures topic diversity before and after. Scripts run in this order; each writes into one `--out` folder.

## Code

**`sourcing/`**
- `source_pages.py`: picks random WARC files from every crawl segment and keeps the pages that pass the quality filters, as text.
- `label_pages.py`: labels each kept page legal or general with the frozen classifier cascade (TF-IDF screen → Laya).

**`filters/`** (used by `source_pages.py`; each page is kept or dropped whole, never edited)
- `url.py`: adult-site URL filter (datatrove's, with the blocklist shared across workers).
- `gopher.py`: Gopher's 21 quality and repetition rules.
- `c4.py`: C4's lorem-ipsum and curly-bracket rules.
- `pipeline.py`: runs every filter in order, stopping at the first rule a page fails.

**`dedup/`**
- `minhash.py`: MinHash building blocks: word 13-grams, signatures, exact Jaccard.
- `fingerprint.py`: fingerprints every kept page, so later steps rarely reread page text.
- `score_pairs.py`: removes exact copies, then finds every near-duplicate pair at Jaccard 0.6 or higher.
- `remove_duplicates.py`: applies each threshold (0.6–0.9): groups connected pages, keeps one per group.

**`topics/`**
- `embed.py`: embeds every kept page (bge-small-en-v1.5, first 512 tokens).
- `fit_topics.py`: fits each bucket's topics with BERTopic once, and groups them into broader levels.
- `assign_topics.py`: gives every page its topic, including pages dedup removes.
- `measure_entropy.py`: the study's numbers: change in effective topics per threshold, against random removal, legal vs general.

**`minhash_pilot/`**: the hand-labeled check run before the full run; see its [README](minhash_pilot/README.md).

## What the corpus contains

### From crawl to kept pages

| Stage | Pages |
|---|---|
| Responses in the 26,776 WARC files | 528,784,667 |
| Not English HTML (out of scope) | −339,029,118 |
| English HTML | 189,755,549 |
| Dropped by URL filter | −2,201,104 |
| Dropped at extraction (empty, timeout, too many tags) | −3,138,698 |
| Dropped by Gopher rules | −83,751,011 |
| Dropped by C4 rules | −1,249,207 |
| **Kept** | **99,415,529** (52% of English HTML) |
| Exact copies removed | −13,113,600 |
| **Before dedup** | **86,301,929** |

Biggest single rules: Gopher word count (40,138,872, pages under 50 words), alphabetic words (15,568,719), duplicate lines (8,781,058), stop words (5,066,669), duplicate 5-grams (3,561,894). Filters run before labeling, so drops aren't split into legal and general.

### Kept pages (after exact copies)

| | General | Legal |
|---|---|---|
| Pages | 86,251,825 | 50,104 |
| Words | 48,385,541,810 | 50,184,523 |
| Median length | 305 words | 419 words |

Share of pages by length:

| Words | General | Legal |
|---|---|---|
| under 100 | 16.4% | 7.0% |
| 100–199 | 20.1% | 17.7% |
| 200–399 | 22.7% | 23.7% |
| 400–799 | 22.4% | 22.0% |
| 800–1,599 | 12.5% | 14.4% |
| 1,600–3,199 | 4.3% | 8.7% |
| 3,200+ | 1.6% | 6.5% |

### After dedup

| Threshold | General pages | General words | Legal pages | Legal words |
|---|---|---|---|---|
| none | 86,251,825 | 48,385,541,810 | 50,104 | 50,184,523 |
| 0.9 | 83,683,096 | 46,265,257,924 (95.6%) | 49,005 | 47,942,573 (95.5%) |
| 0.8 | 81,059,333 | 45,023,640,604 (93.1%) | 48,151 | 46,527,575 (92.7%) |
| 0.7 | 78,688,609 | 43,999,756,308 (90.9%) | 47,511 | 45,633,194 (90.9%) |
| 0.6 | 76,415,319 | 43,127,309,766 (89.1%) | 46,841 | 44,667,208 (89.0%) |

Dedup removes fewer legal pages than general ones, but nearly the same share of words: the legal pages it removes are longer than the average legal page.

### Topics

| | General | Legal |
|---|---|---|
| Fit sample (pages left at 0.6, capped at 1M) | 1,000,000 of 76,415,319 | 46,841 of 46,841 |
| Minimum topic size | 500 | 23 |
| Topics: fine / middle / broad | 218 / 54 / 14 | 283 / 71 / 18 |
| Outliers in the fit (moved to the nearest topic) | 53.5% | 38.5% |

The 5 largest topics in each fit sample. BERTopic describes each topic only by its most typical words; the names are a reading of those words.

| General topic | Top words | Pages |
|---|---|---|
| Online gambling | casino, bonus, games, gambling | 21,343 |
| Christianity | god, jesus, church, christ | 17,204 |
| Skin and hair care | skin, hair, oil, skincare | 13,199 |
| Video games | game, games, players, gaming | 12,898 |
| Recipes | recipe, chicken, ingredients, cheese | 12,602 |

| Legal topic | Top words | Pages |
|---|---|---|
| Court cases (petitioner vs respondent) | vs, petitioner, learned, respondents | 1,161 |
| South Carolina legislature | carolina, south, house, senate | 997 |
| Court motions and rules | motion, party, rule, court | 889 |
| Virginia funds and fiscal years | virginia, year, fund, second | 771 |
| Idaho statute pages (print-friendly) | idaho, friendly, print, website | 746 |

Many legal topics follow a jurisdiction (South Carolina, Virginia, Idaho, Delaware, Vermont) as much as a subject.

## Timing

Measured on HiPerGator on one job of 32 CPUs + 1 RTX 6000. The run: **26,776 files** (24 failed on S3 throttling, not retried), **99,415,529 kept pages** (86,301,929 after exact copies: 86,251,825 general, 50,104 legal).

### Each step

| Step | Time per 1M kept pages | Based on |
|---|---|---|
| Sourcing | ~21 min (0.216 files/s) | live batches 20–268: 26,413 files in 33:58:16 |
| Label / fingerprint / embed | keep up with sourcing | same batches: each finished within ~1 min of sourcing |
| Score pairs | ~7.3 min (not linear: 1.2 min at 1.36M pages) | full run: 12:08:18, `--workers 30` |
| Remove duplicates | ~2.5 s | full run: 0:04:07 |
| Fit topics | 59:29, fixed (general fit capped at 1M pages) | full run, `--workers 28` |
| Assign topics | ~4.7 min | full run: 7:43:46, `--workers 16` (9.2 GB each, 128 GB) |
| Measure entropy | ~1.5 s | full run: 0:02:32, `--threads 32` |

### Blocks

Each block needs the previous one finished; steps inside a block run at the same time.

| Block | Steps | Time, 99.4M pages |
|---|---|---|
| 1 | sourcing + label + fingerprint + embed (live, `--watch`) | **33:58:16** |
| 2 | score pairs | **12:08:18** |
| 3 | remove duplicates | **0:04:07** |
| 4 | fit topics (general + legal) | **1:02:39** |
| 5 | assign topics (general + legal) | **7:43:46** |
| 6 | measure entropy | **0:02:32** |
| | **total** | **~55 h** |

Block 2 ran 6× its linear estimate: candidate pairs grew faster than pages (297,829 at 1.36M pages → 57,504,285 at 99.4M).

**Formula** (fits this run's size; underestimates above ~100M because of block 2): **hours ≈ 0.54 × M + 1.0** (M = millions of kept pages), or **hours ≈ 1.07 × L + 1.0** (L = thousands of legal pages after exact copies).

### Live run times

The real run for the paper, built up in batches into `data/source` with `sbatch jobs/block1.sh <per-segment>` (sourcing `--workers 56`, TF-IDF 1, fingerprint 3 (1 in batch 20), embed bge, all live). Each batch only does the files the last one didn't.

| Batch | Date | New files | Total files | Sourcing | Rate |
|---|---|---|---|---|---|
| `--per-segment 10` | 2026-10-07 | 363 | 363 | timing run | 0.115 files/s |
| `--per-segment 20` | 2026-10-08 | 1,637 | 2,000 | 2:05:29 | 0.217 files/s |
| `--per-segment 60` | 2026-10-08 | 3,999 of 4,000 | 5,999 | 5:03:24 | 0.220 files/s |
| `--per-segment 120` | 2026-10-09 | 6,000 of 6,001 | 11,999 | 7:57:50 | 0.209 files/s |
| `--per-segment 180` | 2026-10-10 | 6,000 of 6,001 | 17,999 | 7:34:12 | 0.220 files/s |
| `--per-segment 220` | 2026-10-10 | 3,959 of 4,001 | 21,958 | 5:01:51 | 0.219 files/s |
| `--per-segment 268` | 2026-10-12 | 4,818 of 4,842 | 26,776 | 6:15:30 | 0.214 files/s |

Label, fingerprint and embed finished within ~1 min of sourcing in every batch except 20 (fingerprint at 1 worker, 28 min late).

### Why score pairs uses first-page pairing

The old every-pair mode (`--all-pairs`) compared every two pages sharing a band value, and its pairs grew with the **square** of the pages. Measured on the old run by taking random subsets of the 363 files and counting only pairs inside each subset (99.6% of pairs are between different files):

| files | pages | pairs >= 0.6 | pairs >= 0.9 |
|---|---|---|---|
| 12 | 42,378 | 3,302 | 268 |
| 23 | 81,248 | 12,446 | 1,041 |
| 45 | 158,982 | 47,007 | 3,810 |
| 91 | 321,233 | 187,259 | 14,771 |
| 181 | 639,460 | 743,796 | 61,678 |
| 363 | 1,281,759 | 2,999,834 | 239,587 |

Pairs ≈ pages^2.0 at both thresholds, so 60M would have meant ~1,940× the pairs (~5.8B) and ~26 h. First-page pairing (the default now) pairs each page only with its band group's first page: at most one pair per band (32) per page, so pairs grow evenly with pages.

| mode | candidates | pairs >= 0.6 | took | 60M |
|---|---|---|---|---|
| every pair (old, `--all-pairs`) | 6,756,333 | 2,999,834 | 0:01:51 (14 workers) | ~26 h |
| first page (now) | 297,829 | 155,788 | 0:01:36 (30 workers) | ~1.0 h |

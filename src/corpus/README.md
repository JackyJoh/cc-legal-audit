# corpus

Code overview: to do.

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
| Measure entropy | not measured yet | |

### Blocks

Each block needs the previous one finished; steps inside a block run at the same time.

| Block | Steps | Time, 99.4M pages |
|---|---|---|
| 1 | sourcing + label + fingerprint + embed (live, `--watch`) | **33:58:16** |
| 2 | score pairs | **12:08:18** |
| 3 | remove duplicates | **0:04:07** |
| 4 | fit topics (general + legal) | **1:02:39** |
| 5 | assign topics (general + legal) | **7:43:46** |
| 6 | measure entropy | not run yet |
| | **total so far** | **~55 h** |

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

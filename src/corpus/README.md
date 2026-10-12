# corpus

Code overview: to do.

## Timing

Measured on HiPerGator on one job of 32 CPUs + 1 RTX 6000. The run: **26,776 files** (24 failed on S3 throttling, not retried), **~99.4M kept pages** and **~53.5k legal pages** (61% counted from label logs, the rest at the measured 3,713 pages per file and 0.054% legal).

### Each step

| Step | Time per 1M kept pages | Based on |
|---|---|---|
| Sourcing | ~21 min (0.216 files/s) | live batches 20–268: 26,413 files in 33:58:16 |
| Label / fingerprint / embed | keep up with sourcing | same batches: each finished within ~1 min of sourcing |
| Score pairs | ~1.2 min | `-N 10`: 1,357,321 pages in 1:36, `--workers 30` |
| Remove duplicates | ~1.5 s | `-N 10`: 0:02 |
| Fit topics | 48 min, fixed (general fit capped at 1M pages) | `-N 10`: 0:48:00, `--workers 28` |
| Assign topics | ~4.3 min | `-N 10`: 5:53, `--workers 16` (6.6 GB each, 128 GB) |
| Measure entropy | ~13 s | `-N 10`: 0:17 |

### Blocks

Each block needs the previous one finished; steps inside a block run at the same time.

| Block | Steps | Time, ~99.4M pages |
|---|---|---|
| 1 | sourcing + label + fingerprint + embed (live, `--watch`) | **~34 h** (measured) |
| 2 | score pairs | **~2.0 h** |
| 3 | remove duplicates | **~2.5 min** |
| 4 | fit topics (general + legal) | **~48 min** |
| 5 | assign topics (general + legal) | **~7.2 h** |
| 6 | measure entropy | **~21 min** |
| | **total** | **~44 h** |

**Formula:** **hours ≈ 0.44 × M + 0.8** (M = millions of kept pages), or **hours ≈ 0.82 × L + 0.8** (L = thousands of legal pages).

**Time so far:** block 1 took 33:58:16 over six batches (2026-10-08 → 10-12), plus the `-N 10` timing run; blocks 2–6 not run yet.

### Live run times

The real run for the paper, built up in batches into `data/source` with `sbatch block1.sh <per-segment>` (sourcing `--workers 56`, TF-IDF 1, fingerprint 3 (1 in batch 20), embed bge, all live). Each batch only does the files the last one didn't.

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

# corpus

Code overview: to do.

## Timing

Measured on HiPerGator, 2026-10-07, on the `--per-segment 10` run (`data/source_n10`: 363 files, 1,357,321 kept pages). 60M kept pages = 60M / 3,739 per file = **16,046 files** (`--per-segment 161`).

### Each step alone

| Step | Measured setup | Measured | Rate | 60M alone, 64 CPUs |
|---|---|---|---|---|
| Sourcing | 32 CPUs, `--workers 64` | 256/1000 files at 41:48 (+64 half done) | 0.115 files/s | **19.4 h** (38.8 h at 32 CPUs, halved) |
| Label | RTX PRO 6000, `--screen-workers 4 --batch-size 128` | 1,357,321 pages, `took 0:04:27` | 5,084/s (TF-IDF 2,674/s each, Laya 145/s) | **2.5 h** (Laya-bound: 60M × 2.15% / 145/s; TF-IDF on 60 workers ~6 min) |
| Fingerprint | `--workers 11` | 1,357,321 pages, `took 0:02:08` | 10,604/s (964/s per worker) | **16 min** (64 workers) |
| Embed (bge) | RTX PRO 6000, `--batch-size 256` | 1,232,192 pages, `took 0:13:13` | 1,554/s | **10.7 h** (GPU-bound) |
| Score pairs | 32 CPUs, `--workers 30` | 1,357,321 pages, `took 0:01:36` (load 1:10, bands 0:05, scores 0:19); 297,829 candidates, 155,788 pairs >= 0.6 | | **~1.0 h** (load and bands don't use more CPUs: 51 + 4 min; scores 14 min → 7 min) |
| Remove duplicates | one process | `took 0:00:02` | | **~1.5 min** (one process) |
| Fit topics | general: 32 CPUs, `--workers 28`; legal: `--workers 2` | general: 1,000,000 pages (capped), `took 0:48:00` (UMAP 32:31, HDBSCAN 8:00, topic words 5:31). legal: 697 pages, `took 0:01:09` | | **~28 min** (UMAP 16 + HDBSCAN 4, halved; topic words 5.5 and the rest 2, unchanged; capped at 1M at any scale) |
| Assign topics | general: 32 CPUs / 128 GB, `--workers 16`, `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`; legal: `--workers 2` | general: 1,281,061 pages, `took 0:05:53`, 6.6 GB per worker (28 workers ran out of memory). legal: 698 pages, `took 0:01:38` | general: 3,629/s (227/s per worker) | **2.2 h** (5:53 × 44 = 4.3 h at 16 workers, halved at 32; needs ~256 GB) |
| Measure entropy | one process | `took ~0:00:17` | | **~12 min** (one process) |

Fit and assign run general and legal at the same time; general always takes longer, so its time is the step's time.

64-CPU times assume the parts that use several CPUs run 2× faster than at 32 CPUs; one-process parts are unchanged. Not measured at 64.

### Blocks

Each block needs the previous one finished; steps inside a block run at the same time.

| Block | Steps | 60M time, 64 CPUs |
|---|---|---|
| 1 | sourcing + label + fingerprint + embed (live, `--watch`) | **~22.2 h** |
| 2 | score pairs | **~1.0 h** |
| 3 | remove duplicates | **~1.5 min** |
| 4 | fit topics (general + legal) | **~28 min** |
| 5 | assign topics (general + legal) | **~2.2 h** (32 workers / 256 GB) |
| 6 | measure entropy | **~12 min** |
| | **total** | **~26 h** |

**Formula** (M = millions of kept pages, 64 CPUs): **hours ≈ 0.43 × M + 0.47**

Every block grows evenly with pages except fit, which is a fixed ~28 min (0.47 h) because the general fit is capped at 1M pages. Per million pages: block 1 0.370 h, score 0.018 h, assign 0.036 h, measure 0.004 h, remove 0.0004 h (sum 0.43 h). Holds for M above ~1.1 (below that the general fit has fewer than 1M pages and is faster).

**By legal pages** (L = thousands of legal pages): **hours ≈ 0.83 × L + 0.47**. Legal is 701 / 1,357,321 = 0.052% of kept pages (516 per million), so 1k legal pages takes ~1.94M kept pages → 0.43 × 1.94 = 0.83 h. E.g. 10k legal ≈ 8.8 h; 31k legal (60M pages) ≈ 26 h.

**For fun, the whole crawl:** CC-MAIN-2026-12 has 100 segments × 1,000 = 100,000 WARC files (`--per-segment 1000`) × 3,739 kept per file ≈ **374M kept pages** → 0.43 × 374 + 0.47 ≈ **161 h ≈ 6.7 days** on one 64-CPU + RTX PRO 6000 job (138 h of it sourcing), with ~194k legal pages at 0.052%.

**Block 1 fit**, at 64 CPUs + 1 RTX PRO 6000 (Laya and embed share it). Sourcing makes ~750 kept pages/s; everything else keeps up, so the block takes as long as sourcing.

| Step | CPUs | Can handle | vs. ~750/s |
|---|---|---|---|
| Sourcing | 56 (`--workers 112`) | ~750/s | sets the pace |
| TF-IDF | 2 (`--screen-workers 2`) | 5,348/s | 7× |
| Laya | 1 + GPU | 109/s (shared GPU) vs. 16/s needed | 6.7× |
| Fingerprint | 3 (`--workers 3`) | 2,892/s | 3.9× |
| Embed | 1 + GPU | ~1,250/s (shared GPU) | 1.7× |
| Main loops | 1 | | |

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

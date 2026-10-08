# corpus

Code overview: to do.

## Timing

Measured on HiPerGator, 2026-10-07, on the `--per-segment 10` run (`data/source_n10`: 363 files, 1,357,321 kept pages). 60M kept pages = 60M / 3,739 per file = **16,046 files** (`--per-segment 161`).

### Each step alone

| Step | Measured setup | Measured | Rate | 60M alone |
|---|---|---|---|---|
| Sourcing | 32 CPUs, `--workers 64` | 256/1000 files at 41:48 (+64 half done) | 0.115 files/s | 38.8 h (32 CPUs), **22.2 h** (56 CPUs) |
| Label | RTX PRO 6000, `--screen-workers 4 --batch-size 128` | 1,357,321 pages, `took 0:04:27` | 5,084/s (TF-IDF 2,674/s each, Laya 145/s) | **3.3 h** |
| Fingerprint | `--workers 11` | 1,357,321 pages, `took 0:02:08` | 10,604/s (964/s per worker) | 1.6 h (11), **5.8 h** (3) |
| Embed (bge) | RTX PRO 6000, `--batch-size 256` | 1,232,192 pages, `took 0:13:13` | 1,554/s | **10.7 h** |
| Score pairs | `--workers 14`, old every-pair mode | 1,357,321 pages, `took 0:01:51` (load 1:03, bands 0:26, scores 0:21); 3.0M pairs >= 0.6 | | not measured (first-page mode not run yet) |
| Remove duplicates | one process, old every-pair mode | `took 0:00:02` | | not measured (first-page mode not run yet) |
| Fit topics | general: 32 CPUs, `--workers 28`; legal: `--workers 2` | general: 1,000,000 pages (capped), `took 0:48:00` (UMAP 32:31, HDBSCAN 8:00, topic words 5:31). legal: 697 pages, `took 0:01:09` | | **~48 min** (general capped at 1M at any scale) |
| Assign topics | general: 32 CPUs / 128 GB, `--workers 16`, `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`; legal: `--workers 2` | general: 1,281,061 pages, `took 0:05:53`, 6.6 GB per worker (28 workers ran out of memory). legal: 698 pages, `took 0:01:38` | general: 3,629/s (227/s per worker) | **4.6 h** (16 workers) |
| Measure entropy | one process | `took ~0:00:17` | | **~12 min** (×44) |

Fit and assign run general and legal at the same time; general always takes longer, so its time is the step's time.

Score pairs / remove duplicates note: pairs grow with the **square** of the pages. Measured by taking random subsets of the 363 files and counting only pairs inside each subset (99.6% of pairs are between different files):

| files | pages | pairs >= 0.6 | pairs >= 0.9 |
|---|---|---|---|
| 12 | 42,378 | 3,302 | 268 |
| 23 | 81,248 | 12,446 | 1,041 |
| 45 | 158,982 | 47,007 | 3,810 |
| 91 | 321,233 | 187,259 | 14,771 |
| 181 | 639,460 | 743,796 | 61,678 |
| 363 | 1,281,759 | 2,999,834 | 239,587 |

Pairs ≈ pages^2.0 at both thresholds. In the old every-pair mode, 60M would mean ~1,940× the pairs (~5.8B, ~26 h for score pairs). So score pairs now pairs each page only with its band group's first page; `--all-pairs` keeps the old mode for measuring what the new one misses.

### Blocks

Each block needs the previous one finished; steps inside a block run at the same time.

| Block | Steps | 60M time |
|---|---|---|
| 1 | sourcing + label + fingerprint + embed (live, `--watch`) | **~22.2 h** |
| 2 | score pairs | not measured |
| 3 | remove duplicates | not measured |
| 4 | fit topics (general + legal) | **~48 min** |
| 5 | assign topics (general + legal) | **~4.6 h** (16 workers / 128 GB) |
| 6 | measure entropy | **~12 min** |

**Block 1 fit**, at 64 CPUs + 1 RTX PRO 6000 (Laya and embed share it). Sourcing makes ~750 kept pages/s; everything else keeps up, so the block takes as long as sourcing.

| Step | CPUs | Can handle | vs. ~750/s |
|---|---|---|---|
| Sourcing | 56 (`--workers 112`) | ~750/s | sets the pace |
| TF-IDF | 2 (`--screen-workers 2`) | 5,348/s | 7× |
| Laya | 1 + GPU | 109/s (shared GPU) vs. 16/s needed | 6.7× |
| Fingerprint | 3 (`--workers 3`) | 2,892/s | 3.9× |
| Embed | 1 + GPU | ~1,250/s (shared GPU) | 1.7× |
| Main loops | 1 | | |
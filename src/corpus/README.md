# corpus

Code overview: to do.

## Timing

Measured on HiPerGator, 2026-10-07, on the `--per-segment 10` run (`data/source_n10`: 363 files, 1,357,321 kept pages). 60M kept pages = 60M / 3,739 per file = **16,046 files** (`--per-segment 161`).

### Each step alone

| Step | Measured setup | Measured | Rate | 60M alone |
|---|---|---|---|---|
| Sourcing | 32 CPUs, `--workers 64` | 256/1000 files at 41:48 (+64 half done) | 0.115 files/s | 38.8 h (32 CPUs), **22.2 h** (56 CPUs) |
| Label | RTX PRO 6000, `--screen-workers 4 --batch-size 128` | 1,357,321 pages, `took 0:04:27`; 29,216 to Laya, 701 legal | 5,084/s (TF-IDF 2,674/s each, Laya 145/s) | **3.3 h** |
| Fingerprint | `--workers 11` | 1,357,321 pages, `took 0:02:08` | 10,604/s (964/s per worker) | 1.6 h (11), **5.8 h** (3) |
| Embed (bge) | RTX PRO 6000, `--batch-size 256` | 1,232,192 pages, `took 0:13:13` | 1,554/s | **10.7 h** |
| Score pairs | `--workers 14` | 1,357,321 pages, `took 0:01:51` (load 1:03, bands 0:26, scores 0:21); 3.0M pairs >= 0.6 | | **~26 h** (see note) |
| Remove duplicates | one process | `took 0:00:02` | | **~1 h** (see note) |
| Fit topics | | not measured yet | | |
| Assign topics | | not measured yet | | |
| Measure entropy | | not measured yet | | |

Score pairs / remove duplicates note: pairs grow with the **square** of the pages. Measured by taking random subsets of the 363 files and counting only pairs inside each subset (99.6% of pairs are between different files):

| files | pages | pairs >= 0.6 | pairs >= 0.9 |
|---|---|---|---|
| 12 | 42,378 | 3,302 | 268 |
| 23 | 81,248 | 12,446 | 1,041 |
| 45 | 158,982 | 47,007 | 3,810 |
| 91 | 321,233 | 187,259 | 14,771 |
| 181 | 639,460 | 743,796 | 61,678 |
| 363 | 1,281,759 | 2,999,834 | 239,587 |

Pairs ≈ pages^2.0 at both thresholds. 60M is 44× the pages, so ~1,940× the pairs: **~5.8B pairs >= 0.6 (~116 GB in pairs.npy)**. Score pairs: load 46 min + (bands 26 s + scores 21 s) × 1,940 ≈ 26 h. Remove duplicates: 2 s × 1,940 ≈ 1 h, holding every pair in memory. The biggest band group (1,130 now) would reach ~50k, past `--max-group` (20k), so score pairs would stop as written.

### Blocks

Each block needs the previous one finished; steps inside a block run at the same time.

| Block | Steps | 60M time |
|---|---|---|
| 1 | sourcing + label + fingerprint + embed (live, `--watch`) | **~22.2 h** |
| 2 | score pairs | **~26 h** |
| 3 | remove duplicates | **~1 h** |
| 4 | fit topics (general + legal) | not measured |
| 5 | assign topics (general + legal) | not measured |
| 6 | measure entropy | not measured |

**Block 1 fit**, at 64 CPUs + 1 RTX PRO 6000 (Laya and embed share it). Sourcing makes ~750 kept pages/s; everything else keeps up, so the block takes as long as sourcing.

| Step | CPUs | Can handle | vs. ~750/s |
|---|---|---|---|
| Sourcing | 56 (`--workers 112`) | ~750/s | sets the pace |
| TF-IDF | 2 (`--screen-workers 2`) | 5,348/s | 7× |
| Laya | 1 + GPU | 109/s (shared GPU) vs. 16/s needed | 6.7× |
| Fingerprint | 3 (`--workers 3`) | 2,892/s | 3.9× |
| Embed | 1 + GPU | ~1,250/s (shared GPU) | 1.7× |
| Main loops | 1 | | |
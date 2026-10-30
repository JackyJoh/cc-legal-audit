"""
Picks the held-out test pages (never trained on) and the pages to drop from
the 0.9 training data so the 0.9 model never sees a near-copy of a test page.

Test pages: N_GENERAL general + N_LEGAL legal pages, picked at random (SEED)
from pages kept at 0.6. A near-copy is a page that 0.6 dedup removed because
it was too similar to a test page. 0.9 dedup is looser and keeps some of them;
those kept ones go in the 0.9 drop list.

Runs in the project environment, from the repo folder.

Usage:
  python src/training/data/heldout.py

Prints: pages kept at 0.6, test pages picked, near-copies found, and how many
near-copies 0.9 kept.
"""
import numpy as np
import os

N_GENERAL = 50_000
N_LEGAL = 2_000
SEED = 42

# paths are from the repo folder
pages = np.load("data/source/dedup/pages.npy")            # id, bucket, file, words
removed_06 = np.load("data/source/dedup/removed_0.6.npy")  # id, kept_id
removed_09 = np.load("data/source/dedup/removed_0.9.npy")  # id, kept_id

page_ids = pages["id"]
buckets = pages["bucket"]   # 0 = general, 1 = legal

# pages kept at 0.6 = pages not in removed_0.6
kept_06 = ~np.isin(page_ids, removed_06["id"])
general_kept = page_ids[kept_06 & (buckets == 0)]
legal_kept = page_ids[kept_06 & (buckets == 1)]
print(f"Kept at 0.6: {len(general_kept):,} general, {len(legal_kept):,} legal")

# pick the test pages at random
rng = np.random.default_rng(SEED)
general_test = rng.choice(general_kept, size=N_GENERAL, replace=False)
legal_test = rng.choice(legal_kept, size=N_LEGAL, replace=False)
test_ids = np.concatenate([general_test, legal_test])
print(f"Test pages: {len(general_test):,} general, {len(legal_test):,} legal")

# near-copies = pages removed at 0.6 whose kept_id is a test page
is_near_copy = np.isin(removed_06["kept_id"], test_ids)
near_copy_ids = removed_06["id"][is_near_copy]

# drop from 0.9 training = near-copies that 0.9 didn't remove
drop_09_ids = near_copy_ids[~np.isin(near_copy_ids, removed_09["id"])]
print(f"Near-copies of test pages: {len(near_copy_ids):,}")
print(f"Near-copies kept at 0.9 (drop from 0.9 training): {len(drop_09_ids):,}")

# store pages
os.makedirs("data/source/training", exist_ok=True)
np.save("data/source/training/heldout_pages.npy", test_ids)
np.save("data/source/training/drop_09_pages.npy", drop_09_ids)
import numpy as np

N_GENERAL = 50_000
N_LEGAL = 2_000
SEED = 42

# load the removed pages from removed_0.6.npy
# paths are from the repo folder (run it from there)
removed_pages = np.load("data/source/dedup/removed_0.6.npy")["id"]
all_pages = np.load("data/source/dedup/pages.npy")
pages = all_pages["id"]
buckets = all_pages["bucket"]

# find pages kept (pages with id not in removed_pages)
kept = ~np.isin(pages, removed_pages)

# split by bucket (0 = general, 1 = legal)
general_pages = pages[kept & (buckets == 0)]
legal_pages = pages[kept & (buckets == 1)]
print(f"Kept at 0.6: {len(general_pages):,} general, {len(legal_pages):,} legal")

# pick at random seed
rng = np.random.default_rng(SEED)
general_sample = rng.choice(general_pages, size=N_GENERAL, replace=False)
legal_sample = rng.choice(legal_pages, size=N_LEGAL, replace=False)

# print the number of sampled pages
print(f"General sample size: {len(general_sample)}")
print(f"Legal sample size: {len(legal_sample)}")


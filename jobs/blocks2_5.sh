#!/bin/bash
#SBATCH --job-name=blocks2_5
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=log/blocks2_5_%j.log

# blocks 2-5 on everything block 1 made, one after another, CPU only:
# score pairs -> remove duplicates -> fit topics -> assign topics.
# Measure entropy (block 6) is left to run by hand. Stops at the first step that fails.
cd /orange/ufdatastudios/$USER/cc-legal-audit
module load python/3.12
set -euo pipefail               # after module load, whose own scripts can trip -u

# block 2
.venv/bin/python src/corpus/dedup/score_pairs.py --out data/source --workers 30 > log/score.log 2>&1

# block 3
.venv/bin/python src/corpus/dedup/remove_duplicates.py --out data/source > log/remove.log 2>&1

# block 4: general and legal at the same time
~/venvs/laya/bin/python src/corpus/topics/fit_topics.py --out data/source --bucket general --workers 28 > log/fit_general.log 2>&1 &
G=$!
~/venvs/laya/bin/python src/corpus/topics/fit_topics.py --out data/source --bucket legal --workers 2 > log/fit_legal.log 2>&1 &
L=$!
wait $G
wait $L

# block 5: one thread per worker; each general worker holds a 6.6 GB model copy, so 16 fit in 128 GB
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
~/venvs/laya/bin/python src/corpus/topics/assign_topics.py --out data/source --bucket general --workers 16 > log/assign_general.log 2>&1 &
G=$!
~/venvs/laya/bin/python src/corpus/topics/assign_topics.py --out data/source --bucket legal --workers 2 > log/assign_legal.log 2>&1 &
L=$!
wait $G
wait $L
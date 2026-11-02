#!/bin/bash
# one training run: sbatch jobs/train.sh <threshold> <seed>, e.g. sbatch jobs/train.sh 0.6 1
# quick check first: sbatch --time=1:00:00 jobs/train.sh 0.6 1 --test
#SBATCH --job-name=train
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --partition=hpg-b200
#SBATCH --time=6-00:00:00
#SBATCH --output=log/train_%j.log

cd /orange/ufdatastudios/$USER/cc-legal-audit
module load python/3.12
export WANDB_PROJECT=ccaudit-dedup

# copy this run's training set to the node's own disk and train from the copy;
# reading it from /orange (shared over the network) left the GPU idle while it
# waited for data. $TMPDIR is this job's folder there, deleted when the job ends.
# If the copy fails (for example the disk is full) the job stops, so it never
# trains on a cut-off file
start=$(date +%s)
cp data/source/training/train_$1.bin $TMPDIR/ || exit 1
echo "copied train_$1.bin to $TMPDIR in $(( $(date +%s) - start )) s"

~/venvs/train/bin/python -u src/training/train/train.py --threshold $1 --seed $2 \
    --bin $TMPDIR/train_$1.bin ${@:3}

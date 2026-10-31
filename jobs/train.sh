#!/bin/bash
# one training run: sbatch jobs/train.sh <threshold> <seed>, e.g. sbatch jobs/train.sh 0.6 1
# quick check first: sbatch --time=0:30:00 jobs/train.sh 0.6 1 --test
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

~/venvs/train/bin/python -u src/training/train/train.py --threshold $1 --seed $2 ${@:3}

#!/bin/bash
# every test on one model: sbatch jobs/eval.sh <model folder>
# e.g. sbatch jobs/eval.sh data/source/training/runs/0.9_seed1/final
#SBATCH --job-name=eval
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-turin
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --output=log/eval_%j.log

cd /orange/ufdatastudios/$USER/cc-legal-audit
module load python/3.12

~/venvs/eval/bin/python -u src/training/eval/evaluate.py --model $1 --test benchmarks --batch-size 64
~/venvs/eval/bin/python -u src/training/eval/evaluate.py --model $1 --test perplexity --batch-size 16
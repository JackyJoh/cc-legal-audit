#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=log/tokenize_%j.log

cd /orange/ufdatastudios/$USER/cc-legal-audit
module load python/3.12

~/venvs/train/bin/python -u src/training/data/tokenize_pages.py --workers 32
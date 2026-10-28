#!/bin/bash
#SBATCH --job-name=throughput
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --partition=hpg-b200
#SBATCH --time=1:00:00
#SBATCH --output=log/throughput_%j.log

cd /orange/ufdatastudios/$USER/cc-legal-audit
module load python/3.12

~/venvs/train/bin/python -u src/training/train/throughput.py
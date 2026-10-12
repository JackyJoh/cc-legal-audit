#!/bin/bash
#SBATCH --job-name=block1
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=log/block1_%j.log

PER=${1:?give files per segment, e.g. sbatch jobs/block1.sh 20}
cd /orange/ufdatastudios/jjohannessen/cc-legal-audit
module load python/3.12
rm -f data/source/source.done

.venv/bin/python src/corpus/sourcing/source_pages.py --per-segment $PER --out data/source --workers 56 > log/source_$PER.log 2>&1 &
~/venvs/laya/bin/python src/corpus/sourcing/label_pages.py --out data/source --screen-workers 1 --batch-size 128 --watch > log/label_$PER.log 2>&1 &
.venv/bin/python src/corpus/dedup/fingerprint.py --out data/source --workers 3 --watch > log/fingerprint_$PER.log 2>&1 &
~/venvs/laya/bin/python src/corpus/topics/embed.py --out data/source --watch > log/embed_$PER.log 2>&1 &
wait
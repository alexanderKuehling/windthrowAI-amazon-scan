#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=WTAI_merge
#SBATCH --output=slurm_logs/merge/%j.out
#SBATCH --error=slurm_logs/merge/%j.err
#SBATCH --kill-on-invalid-dep=yes

set -euo pipefail

mkdir -p slurm_logs

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv

# Exits non-zero — cancelling the chained export (03) — on missing tiles, too
# many download errors or a positive count that moved by more than 50%; see
# the script's docstring. slurm/resume_scan.sh forwards --allow-errors /
# --accept-delta here.
python -u scripts/merge_results.py "$@"

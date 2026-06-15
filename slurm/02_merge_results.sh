#!/bin/bash
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=WTAI_merge
#SBATCH --output=slurm_logs/merge-%j.out
#SBATCH --error=slurm_logs/merge-%j.err

mkdir -p slurm_logs

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv

python -u scripts/merge_results.py

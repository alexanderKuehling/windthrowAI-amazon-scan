#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=WTAI_export
#SBATCH --output=slurm_logs/export-%j.out
#SBATCH --error=slurm_logs/export-%j.err
#SBATCH --mail-type=FAIL

mkdir -p slurm_logs

ml proxy

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv

python -u scripts/export_for_frontend.py

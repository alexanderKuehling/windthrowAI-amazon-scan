#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=WTAI_grid
#SBATCH --output=slurm_logs/grid-%j.out
#SBATCH --error=slurm_logs/grid-%j.err

mkdir -p slurm_logs

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv

python -u scripts/generate_grid.py

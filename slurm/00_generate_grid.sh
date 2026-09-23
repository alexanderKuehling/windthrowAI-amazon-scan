#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=WTAI_grid
#SBATCH --output=slurm_logs/grid/%j.out
#SBATCH --error=slurm_logs/grid/%j.err

set -euo pipefail

mkdir -p slurm_logs

ml proxy   # network access for resolve_model_versions.py (Hugging Face Hub)

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv

python -u scripts/generate_grid.py

# Pins the model checkpoints for this whole scan (writes model_versions.json),
# so every 01_inference_array.sh task uses the same revision. See src/hub.py.
python -u scripts/resolve_model_versions.py

#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=WTAI_infer
#SBATCH --output=slurm_logs/infer/%A_%a.out
#SBATCH --error=slurm_logs/infer/%A_%a.err
#SBATCH --array=0-115          # adjust 
#SBATCH --mail-type=FAIL

set -euo pipefail

mkdir -p slurm_logs

module load cuda/12.1
ml proxy

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv


# Default: pull the checkpoints pinned in model_versions.json (written by
# resolve_model_versions.py in slurm/00_generate_grid.sh) — the backbone is
# auto-detected from the checkpoint, no manual pairing needed. To test a local
# experimental checkpoint instead, pass --segmenter-path/--classifier-path
# (and --backbone if the checkpoint has none embedded), same as resegment.py.
python -u scripts/run_inference.py \
    --job-idx "$SLURM_ARRAY_TASK_ID" \
    --date-from "2024-06-01" \
    --date-to   "2024-09-30"

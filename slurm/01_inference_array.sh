#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=WTAI_infer
#SBATCH --output=slurm_logs/infer-%A_%a.out
#SBATCH --error=slurm_logs/infer-%A_%a.err
#SBATCH --array=0-119          # adjust 
#SBATCH --mail-type=FAIL


mkdir -p slurm_logs

module load cuda/12.1
ml proxy

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv


python -u scripts/run_inference.py \
    --job-idx "$SLURM_ARRAY_TASK_ID" \
    --date-from "2024-06-01" \
    --date-to   "2024-09-30"

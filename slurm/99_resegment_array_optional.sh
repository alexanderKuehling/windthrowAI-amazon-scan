#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=WTAI_reseg
#SBATCH --output=slurm_logs/reseg/%A_%a.out
#SBATCH --error=slurm_logs/reseg/%A_%a.err
#SBATCH --array=0-120%8     # adjust: ceil(num_positives / chunk_size) - 1; keep %8 on GEE (budget, see 01)
#SBATCH --mail-type=FAIL

# Optional, off the standard pipeline: re-segment the tiles already flagged
# positive with a different checkpoint, without a full re-scan. The export step
# takes outputs_resegment/ over outputs/ per tile, so this can also be run on a
# subset — untouched tiles keep their original mask.

mkdir -p slurm_logs

# Optional: the cluster no longer has this module, and PyTorch's wheels bring
# their own CUDA runtime. Must not fail the task under `set -e`.
module load cuda/12.1 2>/dev/null || echo "cuda/12.1 module not available, using PyTorch's bundled CUDA"
ml proxy

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv

python -u scripts/resegment.py \
    --job-idx "$SLURM_ARRAY_TASK_ID" \
    --segmenter-path models/segmenter_v2.pth \
    --backbone resnet34 \
    --chunk-size 500

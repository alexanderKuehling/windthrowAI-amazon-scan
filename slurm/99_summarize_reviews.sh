#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=WTAI_reviewstats
#SBATCH --output=slurm_logs/reviewstats/%j.out
#SBATCH --error=slurm_logs/reviewstats/%j.err
#SBATCH --mail-type=FAIL

# Off the scan pipeline: report how much expert-review signal has accumulated
# in the HF dataset, split into what it can train — tile-level labels for the
# classifier and corrected masks for the segmenter. Read this before starting a
# retraining round; it is what tells you whether there is enough, and how many
# reviewers have seen each tile.
#
#   sbatch slurm/99_summarize_reviews.sh                     # full report
#   sbatch slurm/99_summarize_reviews.sh --fast              # counts only, no download
#   sbatch slurm/99_summarize_reviews.sh --since 2026-08-01  # only recent reviews
#
# Writes outputs_final/reviews/review_summary_per_tile.csv (the label set the
# retraining reads) and review_summary_per_reviewer.csv. Read-only against the
# dataset — it never uploads.

mkdir -p slurm_logs

ml proxy

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv

python -u scripts/summarize_reviews.py "$@"

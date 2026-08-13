#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=WTAI_recompress
#SBATCH --output=slurm_logs/recompress/%j.out
#SBATCH --error=slurm_logs/recompress/%j.err
#SBATCH --mail-type=FAIL

# One-time migration of the inputs/ already published in the HF dataset to the
# compact uint16 encoding (~2.0 MB -> ~0.6 MB per tile). New exports need this
# no more — 03_export_frontend.sh writes that encoding directly.
#
#   sbatch slurm/99_recompress_hf_inputs_one_time.sh            # dry run
#   sbatch slurm/99_recompress_hf_inputs_one_time.sh --upload   # publish
#
# --upload needs a write token: `huggingface-cli login` on the cluster, or
# export HF_TOKEN before submitting.

mkdir -p slurm_logs

ml proxy

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv

python -u scripts/recompress_hf_inputs.py "$@"

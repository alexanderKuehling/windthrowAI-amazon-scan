#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=WTAI_export
#SBATCH --output=slurm_logs/export/%j.out
#SBATCH --error=slurm_logs/export/%j.err
#SBATCH --mail-type=FAIL
#SBATCH --kill-on-invalid-dep=yes

set -euo pipefail

mkdir -p slurm_logs

ml proxy

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate windthrowEnv

# --skip-tiles is permanent: tiles/ is the map overlay of
# windthrowAI-segmentation-FE-hosting, which the review/retrain loop doesn't
# serve (decision D3), and gdal2tiles over z5-13 is the expensive part of a run.
# --skip-inputs was removed: new positive tiles need their review imagery
# uploaded too, or the Expert-Review tab 404s on inputs/{tile}.tif (F7).
python -u scripts/export_for_frontend.py --skip-tiles

#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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

# GDAL's command-line tools (gdalbuildvrt, gdal2tiles.py) — the map overlay
# needs them, rasterio's pip wheel doesn't ship them. If windthrowEnv lacks
# them, use a separate env named `gdal` (once: conda create -n gdal -c
# conda-forge gdal), appended to PATH so windthrowEnv's python stays first.
if ! command -v gdalbuildvrt >/dev/null; then
    GDAL_ENV=$(conda env list | awk '$1 == "gdal" {print $NF}')
    [[ -n "$GDAL_ENV" ]] && export PATH="$PATH:$GDAL_ENV/bin"
fi

# Full export incl. the map overlay (tiles/, gdal2tiles over z5-13, the slow
# part — hence the 6 h walltime, a margin rather than a measurement: an
# export that times out publishes nothing). tiles/ is mirrored, so detections
# a new model dropped also disappear from the map.
# --skip-inputs was removed: new positive tiles need their review imagery
# uploaded too, or the Expert-Review tab 404s on inputs/{tile}.tif (F7).
python -u scripts/export_for_frontend.py "$@"   # e.g. --skip-tiles (slurm/resume_scan.sh forwards it)

#!/bin/bash
# Submit the full 00→01→02→03 scan pipeline as one chained sequence.
#
# This is the deliberate "start a new scan" action — nothing triggers it
# automatically (not even a new model publish in windthrowAI-training). Run it
# by hand from the repo root:
#
#     slurm/run_full_scan.sh
#
# Step 00 (grid + pinned model versions) takes up to ~15 min and this script
# blocks on it (`sbatch --wait`) before it can read the array size — run this
# under tmux/screen or `nohup slurm/run_full_scan.sh & disown`, since an SSH
# drop during those 15 minutes kills the wait before 01 is ever submitted.
#
# Steps 01→02→03 are then chained via `sbatch --dependency=afterok:<jobid>`
# and this script returns immediately after submitting them — SLURM's own
# scheduler carries the rest, you don't need to stay logged in for that part.
#
# The optional checkpoint-review step (99_resegment_array_optional.sh) is
# NOT part of this chain — swapping in a newly validated segmenter stays a
# separate, manually-inserted step between 02 and 03, per slurm/README.md.
#
# A failed array task in 01 does not cancel 02/03 by itself; thanks to
# --kill-on-invalid-dep=yes on those scripts it instead surfaces as a visible
# CANCELLED job in `squeue` rather than sitting silently pending forever. The
# same happens when 02 stops on missing tiles, download errors or a drifted
# positive count. Either way, slurm/resume_scan.sh finishes the scan.

set -euo pipefail
cd "$(dirname "$0")/.."

# SLURM opens each job's --output/--error file before the job starts, so a
# missing log folder (reset.md deletes slurm_logs/) fails the job at once with
# no log at all. The mkdir inside the job scripts runs too late for that.
mkdir -p slurm_logs/grid slurm_logs/infer slurm_logs/merge slurm_logs/export

# Step 03 overwrites inputs/ and masks/ on HF. Reviews from before scan_id
# existed are pinned to the `scan-legacy` tag, so refuse to start a scan whose
# export would destroy what those reviewers saw (one-time tag: LOOP.md).
if ! python scripts/export_for_frontend.py --check-legacy-tag; then
    exit 1
fi

SOURCE=$(python -c "from src.config import DATA_SOURCE; print(DATA_SOURCE)")
echo "Imagery source: $SOURCE (set WINDTHROW_DATA_SOURCE=gee|pc to change)"
# Every array task imports the source's client; a missing package would fail
# all of them hours from now, so fail here instead.
if [[ "$SOURCE" == "pc" ]] && ! python -c "import pystac_client, planetary_computer" 2>/dev/null; then
    echo "Planetary Computer packages missing: pip install -r requirements.txt" >&2
    exit 1
fi

echo "Submitting grid generation + model resolution (00), waiting..."
if ! sbatch --wait slurm/00_generate_grid.sh; then
    echo "step 00 failed — aborting, nothing else submitted" >&2
    exit 1
fi

N=$(python -c "
from src.tile_grid import n_jobs
from src.config import GRID_FILE, TILES_PER_JOB
print(n_jobs(GRID_FILE, TILES_PER_JOB) - 1)
")
# At most this many array tasks at once — on GEE the scan's share of the
# project's concurrent-request quota, on Planetary Computer a bandwidth share
# (see src/config.py and src/sources.py::array_parallel).
PAR=$(python -c "from src.sources import array_parallel; print(array_parallel())")
echo "Array range: 0-$N (max $PAR running)"

INFER_JOBID=$(sbatch --parsable --array=0-"$N"%"$PAR" slurm/01_inference_array.sh)
echo "inference   : $INFER_JOBID (array 0-$N%$PAR)"

MERGE_JOBID=$(sbatch --parsable --dependency=afterok:"$INFER_JOBID" slurm/02_merge_results.sh)
echo "merge       : $MERGE_JOBID"

EXPORT_JOBID=$(sbatch --parsable --dependency=afterok:"$MERGE_JOBID" slurm/03_export_frontend.sh)
echo "export      : $EXPORT_JOBID"

echo
echo "Full scan submitted. Monitor with: squeue -u \$USER"

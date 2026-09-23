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
# CANCELLED job in `squeue` rather than sitting silently pending forever.

set -euo pipefail
cd "$(dirname "$0")/.."

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
echo "Array range: 0-$N"

INFER_JOBID=$(sbatch --parsable --array=0-"$N" slurm/01_inference_array.sh)
echo "inference   : $INFER_JOBID (array 0-$N)"

MERGE_JOBID=$(sbatch --parsable --dependency=afterok:"$INFER_JOBID" slurm/02_merge_results.sh)
echo "merge       : $MERGE_JOBID"

EXPORT_JOBID=$(sbatch --parsable --dependency=afterok:"$MERGE_JOBID" slurm/03_export_frontend.sh)
echo "export      : $EXPORT_JOBID"

echo
echo "Full scan submitted. Monitor with: squeue -u \$USER"

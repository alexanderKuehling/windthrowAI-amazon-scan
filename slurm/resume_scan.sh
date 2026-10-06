#!/bin/bash
# Finish a scan whose merge (02) stopped, or whose inference array (01) had
# failed tasks: resubmit just the array tasks that still miss tiles or hold
# download errors, then chain 02 -> 03 after them again.
#
#     slurm/resume_scan.sh                   # redo unfinished tasks, then merge + export
#     slurm/resume_scan.sh --accept-delta    # ... and accept a >50% positive-count change
#     slurm/resume_scan.sh --allow-errors    # ... and publish without tiles that keep failing
#
# Run from anywhere on the login node, windthrowEnv active. Flags are passed
# to scripts/merge_results.py. A resubmitted task resumes from its results.csv
# (see reset.md), so finished tiles are never downloaded twice. When every task
# is complete it submits just 02 -> 03.

set -euo pipefail
cd "$(dirname "$0")/.."

# SLURM opens each job's --output/--error file before the job starts, so a
# missing log folder (reset.md deletes slurm_logs/) fails the job at once with
# no log at all. The mkdir inside the job scripts runs too late for that.
mkdir -p slurm_logs/infer slurm_logs/merge slurm_logs/export

JOBS=$(python scripts/merge_results.py --list-incomplete)
PAR=$(python -c "from src.sources import array_parallel; print(array_parallel())")

DEP=()
if [[ -n "$JOBS" ]]; then
    INFER_JOBID=$(sbatch --parsable --array="$JOBS"%"$PAR" slurm/01_inference_array.sh)
    echo "inference   : $INFER_JOBID (tasks $JOBS, max $PAR running)"
    DEP=(--dependency=afterok:"$INFER_JOBID")
else
    echo "inference   : every task complete"
fi

MERGE_JOBID=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} slurm/02_merge_results.sh "$@")
echo "merge       : $MERGE_JOBID ${*:-}"

EXPORT_JOBID=$(sbatch --parsable --dependency=afterok:"$MERGE_JOBID" slurm/03_export_frontend.sh)
echo "export      : $EXPORT_JOBID"

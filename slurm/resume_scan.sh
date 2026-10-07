#!/bin/bash
# Finish a scan whose merge (02) stopped, or whose inference array (01) had
# failed tasks: resubmit just the array tasks that still miss tiles or hold
# download errors, then chain 02 -> 03 after them again.
#
#     slurm/resume_scan.sh                   # redo unfinished tasks, then merge + export
#     slurm/resume_scan.sh --accept-delta    # ... and accept a >50% positive-count change
#     slurm/resume_scan.sh --allow-errors    # ... and publish without tiles that keep failing
#     slurm/resume_scan.sh --skip-tiles      # ... and don't refresh the map overlay (export)
#
# Run from anywhere on the login node, windthrowEnv active. --skip-tiles goes
# to the export (scripts/export_for_frontend.py), every other flag to
# scripts/merge_results.py. A resubmitted task resumes from its results.csv
# (see reset.md), so finished tiles are never downloaded twice. When every task
# is complete it submits just 02 -> 03.

set -euo pipefail
cd "$(dirname "$0")/.."

# Logs of this run go to slurm_logs/<step>/<RUN>/, so the 116 inference task
# logs of one attempt sit in one folder: `rm -rf slurm_logs/*/<RUN>` clears it.
# SLURM opens each job's --output/--error file before the job starts and does
# not create folders, so they are made here and passed on the sbatch line
# (the job ids aren't known before submission, hence the timestamp).
RUN=$(date -u +%Y%m%dT%H%M%SZ)
logdir() { mkdir -p "slurm_logs/$1/$RUN" && echo "slurm_logs/$1/$RUN"; }

MERGE_ARGS=(); EXPORT_ARGS=()
for a in "$@"; do
    if [[ "$a" == --skip-tiles ]]; then EXPORT_ARGS+=("$a"); else MERGE_ARGS+=("$a"); fi
done

JOBS=$(python scripts/merge_results.py --list-incomplete)
PAR=$(python -c "from src.sources import array_parallel; print(array_parallel())")

DEP=()
if [[ -n "$JOBS" ]]; then
    L=$(logdir infer)
    INFER_JOBID=$(sbatch --parsable --array="$JOBS"%"$PAR" --output="$L/%A_%a.out" --error="$L/%A_%a.err" \
        slurm/01_inference_array.sh)
    echo "inference   : $INFER_JOBID (tasks $JOBS, max $PAR running)"
    DEP=(--dependency=afterok:"$INFER_JOBID")
else
    echo "inference   : every task complete"
fi

L=$(logdir merge)
MERGE_JOBID=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} --output="$L/%j.out" --error="$L/%j.err" \
    slurm/02_merge_results.sh ${MERGE_ARGS[@]+"${MERGE_ARGS[@]}"})
echo "merge       : $MERGE_JOBID ${MERGE_ARGS[*]:-}"

L=$(logdir export)
EXPORT_JOBID=$(sbatch --parsable --dependency=afterok:"$MERGE_JOBID" --output="$L/%j.out" --error="$L/%j.err" \
    slurm/03_export_frontend.sh ${EXPORT_ARGS[@]+"${EXPORT_ARGS[@]}"})
echo "export      : $EXPORT_JOBID ${EXPORT_ARGS[*]:-}"
echo "logs        : slurm_logs/*/$RUN/"

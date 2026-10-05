# Resetting the scan

How to bring the repo back to a clean state so the pipeline can run from step 00
again. Everything here is a plain shell command — run it from the repo root on
the login node, no conda environment needed.

```bash
cd windthrowAI-amazon-scan
```

## What a run writes

| Path | Written by | Delete for a fresh run? |
|------|-----------|--------------------------|
| `outputs/` | 01 inference — `job_*/results.csv`, `run_meta.json`, `*_mask.tif`, `*_input.tif`, `*_prob.tif` | **yes** |
| `outputs/all_results.csv`, `*.vrt`, `*.geojson` | 02 merge | yes (inside `outputs/`) |
| `outputs_resegment/` | 99 re-segmentation | **yes** |
| `outputs_final/` | 03 export staging — `tiles/`, `inputs/`, `masks/`, `probs/`, `inputs_index.csv`, `scan_meta.json` | **yes** |
| `slurm_logs/` | every job | optional |
| `src/__pycache__`, `scripts/__pycache__` | Python | optional |
| `tile_grid.parquet` | 00 generate grid | **no** — see below |
| `model_versions.json` | 00 (resolve_model_versions.py) | doesn't matter — overwritten fresh on every `00` run, unlike `tile_grid.parquet` it is never meant to be kept |
| `models/`, `gee-windthrowai-secret.json` | you | **never** |

## Start from zero

```bash
rm -rf outputs outputs_resegment outputs_final
rm -rf slurm_logs src/__pycache__ scripts/__pycache__
```

`tile_grid.parquet` is deliberately kept: it is a deterministic function of the
grid parameters in `src/config.py`, so unless those changed it will be rebuilt
identically and you can skip step 00. Delete it only if you changed
`AMAZON_BBOX`, `TILE_SIZE_PX`, `RESOLUTION_M` or `OVERLAP_PX`:

```bash
rm -f tile_grid.parquet     # then re-run 00_generate_grid.sh
```

## Keep the previous scan instead of deleting it

A rename is instant and costs no copy — worth it before a run you might want to
compare against, or when the last scan was expensive to produce:

```bash
STAMP=$(date +%Y%m%dT%H%M%S)
mv outputs           outputs_archive_$STAMP
mv outputs_resegment outputs_resegment_archive_$STAMP
mv outputs_final     outputs_final_archive_$STAMP
```

The archives still count against your disk quota — check with `du -sh outputs*`
before and delete them once the new scan is validated. Nothing in the pipeline
reads `*_archive_*`, so they are inert.

## If `outputs/` is a symlink to scratch

Clear the contents and keep the link, so jobs keep writing to the same place:

```bash
ls -ld outputs                    # confirm it is a symlink first
rm -rf outputs/* outputs/.??*
```

## Why a half-reset is worse than no reset

`scripts/run_inference.py` resumes: on start it reads `outputs/job_XXXX/results.csv`
and skips every `tile_id` already listed — except downloads that failed on a GEE
error, which it retries. That is what makes a failed array task
safe to resubmit — and what makes a leftover directory dangerous.

If a `results.csv` from an **older** run is still in place, those tiles are
treated as done. They keep their old columns, and the new ones (`clf_prob`,
`valid_frac`, the `seg_*` scores, the composite dates) stay empty for them — the
CSV is migrated to the new header with blank fields rather than re-run. You end
up with a result set that is silently part old, part new.

So: either delete `outputs/` completely, or leave it completely alone. Do not
delete individual mask files while keeping the CSV.

Deliberately re-running a subset is fine — delete the whole `job_XXXX/` directory
and resubmit just that index:

```bash
rm -rf outputs/job_0042
sbatch --array=42 slurm/01_inference_array.sh
```

## The published Hugging Face dataset

A local reset does not touch
`alexKlg/windthrowAI-dataset-segmentation_results`. In most cases it does not
need to: step 03 overwrites every file it republishes, and the Expert-Review tab
only shows tiles listed in `inputs_index.csv`, so `inputs/` or `masks/` entries
left over from a previous scan are invisible to reviewers — they only occupy
repo storage.

Clean them out only if you want the storage back, and note that this is
irreversible and affects the live frontend:

```python
from huggingface_hub import HfApi
api = HfApi()   # needs a write token
for folder in ("inputs", "masks", "tiles", "probs"):
    api.delete_folder(folder, repo_id="alexKlg/windthrowAI-dataset-segmentation_results",
                      repo_type="dataset")
```

`reviews/` holds the expert corrections submitted through the frontend. It is
never written by the scan and must not be deleted — it is the only copy of that
work.

## After the reset

```bash
slurm/run_full_scan.sh
```

Chains `00 → 01 → 02 → 03` automatically (waits for `00`, reads the array size
from `tile_grid.parquet`, wires the `--dependency=afterok` chain) — run it
under `tmux`/`screen` since it blocks on step 00 for ~15 min. Spelled out by
hand instead:

```bash
sbatch slurm/00_generate_grid.sh          # regenerates tile_grid.parquet (if deleted) and model_versions.json
# read the printed "--array=0-N" from slurm_logs/grid/*.out

sbatch --array=0-115%8 slurm/01_inference_array.sh
sbatch --dependency=afterok:<infer_jobid> slurm/02_merge_results.sh
sbatch --dependency=afterok:<merge_jobid> slurm/03_export_frontend.sh
```

Check `slurm/README.md` for the full pipeline, and confirm before submitting
step 03 that its flags are the ones you want. By default it runs the full
export (map overlay, review imagery, masks, index); `--skip-tiles` leaves the
map as it is for a quicker run, `--skip-inputs` skips the review imagery.
Step 03 also tags the HF dataset
repo with the export's `scan_id` once every upload succeeds — reviews
submitted afterwards belong to that round.

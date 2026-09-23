# SLURM pipeline

> Operating the review → retrain → rescan loop: see [`LOOP.md`](../../LOOP.md) (lives next to the repos, outside git).

Amazon-wide windthrow scan, from an empty repo to the Hugging Face dataset the
frontends read. Submit every job **from the repo root** — the scripts use paths
relative to it (`scripts/`, `outputs/`, `slurm_logs/`):

```bash
cd windthrowAI-amazon-scan
slurm/run_full_scan.sh
```

That one command chains the whole `00`–`03` sequence (see "Full run" below).
The rest of this doc describes each step individually — useful for running a
single step by hand, debugging, or resubmitting a failed array task.

All jobs activate the `windthrowEnv` conda environment. The GEE-facing and
HF-facing jobs load `ml proxy` for outbound network access.

`00`–`03` are the standard pipeline and run in that order. `99_*` scripts are
exceptions: none of them is part of a normal scan.

To re-run a scan from scratch, clear the previous one first —
[`reset.md`](../reset.md) lists what to delete and what to keep.

| # | Script | Job | GPU | Array | Walltime | Reads | Writes |
|---|--------|-----|-----|-------|----------|-------|--------|
| 00 | `00_generate_grid.sh` | `WTAI_grid` | no | – | 15 min | `src/config.py`, HF model repos | `tile_grid.parquet`, `model_versions.json` |
| 01 | `01_inference_array.sh` | `WTAI_infer` | 2×A100 | yes | 4 h | `tile_grid.parquet` | `outputs/job_*/` |
| 02 | `02_merge_results.sh` | `WTAI_merge` | no | – | 1 h | `outputs/job_*/results.csv` | `outputs/all_results.csv`, `.vrt`, `.geojson` |
| 03 | `03_export_frontend.sh` | `WTAI_export` | no | – | 6 h | `outputs/`, `outputs_resegment/` | `outputs_final/` + HF dataset |
| 99 | `99_resegment_array_optional.sh` | `WTAI_reseg` | 2×A100 | yes | 2 h | `outputs/all_results.csv` | `outputs_resegment/job_*/` |
| 99 | `99_recompress_hf_inputs_one_time.sh` | `WTAI_recompress` | no | – | 3 h | HF dataset `inputs/` | HF dataset `inputs/` |
| 99 | `99_summarize_reviews.sh` | `WTAI_reviewstats` | no | – | 30 min | HF dataset `reviews/` | `outputs_final/reviews/*.csv` |

## The standard pipeline

### 00 — Generate the tile grid

Builds the spatial grid covering the Amazon (256 px tiles at 30 m, Landsat-8
native resolution) and writes `tile_grid.parquet`. Run once; re-run only after
changing the grid parameters in `src/config.py`.

It also runs `scripts/resolve_model_versions.py`, which resolves the current
HEAD commit of both production HF Hub model repos, downloads both checkpoints
once to warm the shared HF cache, and writes `model_versions.json` at the repo
root. Every array task in step 01 reads that file and pulls the same pinned
revision — resolved once per scan, not independently per task, so a model
promoted mid-scan can't produce a run with mixed checkpoints. Unlike
`tile_grid.parquet`, `model_versions.json` is meant to be regenerated on every
scan, not kept — see [`reset.md`](../reset.md).

**Its output tells you the array size for step 01** — it prints the exact flag,
and (if you're not using `slurm/run_full_scan.sh`) you'd read it from here:

```
Total tiles   : <n>
Tiles per job : 2000
SLURM array   : 0–<n/2000-1>  (--array=0-...)
```

### 01 — Inference (array job)

The main GPU pass. Each array task takes `TILES_PER_JOB` (2000) tiles from the
grid, downloads the Landsat-8 dry-season composite for each from Google Earth
Engine, runs the classifier, and segments the tiles it flags positive.

Per task: `outputs/job_{idx:04d}/results.csv` and `run_meta.json`, plus
`{tile_id}_mask.tif`, `{tile_id}_input.tif` (the 7-band model input) and
`{tile_id}_prob.tif` (the segmenter probability map) for positive tiles only.

`results.csv` records the model's **scores**, not just its decisions — see
`src/results.py` for the column list:

| Column | Written for | Why it is worth the disk |
|--------|-------------|---------------------------|
| `clf_prob` | every downloaded tile, negatives included | The classifier gate is the one irreversible step of the scan. With the probability stored, a threshold change is a CSV query; without it, it is a full re-scan. |
| `valid_frac` | every downloaded tile | Share of pixels that survived cloud masking. A detection over a mostly-empty tile is the most common false positive. |
| `n_scenes`, `date_first`, `date_last`, `cloud_mean` | every downloaded tile | Which imagery the composite was built from — needed to date a detection and to reproduce it. |
| `coverage`, `mask_pixels` | positives | Extent of the proposed mask; drives the Expert-Review coverage filter. |
| `seg_mean_prob`, `seg_mean_prob_in_mask`, `seg_max_prob` | positives | How confident the segmenter was, not just where it crossed 0.5. |
| `prob_path` | positives | The `_prob.tif` (uint8, probability = DN/255, ~25 kB) that makes the segmentation threshold re-tunable offline. |

`--clf-threshold` / `--seg-threshold` set the gates; both scores are recorded
regardless of where the gates sit. `--no-prob-maps` skips the probability
rasters — which also gives up post-hoc re-thresholding.

Resuming is by `tile_id` against the existing `results.csv`, and a CSV from an
older run is migrated to the current header on open (old rows padded with empty
fields, never re-run). See [`reset.md`](../reset.md) before re-running a scan
over a directory that already holds one.

Set `--array` to the range printed by step 00 (`slurm/run_full_scan.sh` does
this for you). By default the checkpoints come from the pinned HF pull —
`model_versions.json`, written by step 00 — and the segmenter's encoder is
auto-detected from the checkpoint, so there's no manual pairing to get wrong.
To test a local experimental checkpoint instead, pass `--segmenter-path`/
`--classifier-path` (and `--backbone` if that checkpoint has no embedded
backbone) directly in the script, the same override pattern `resegment.py`
uses. Dates default to the dry season, `2024-06-01`–`2024-09-30`.

Needs GEE credentials: `gee-windthrowai-secret.json` in the repo root
(or `--key-path`).

### 02 — Merge results

Concatenates every `outputs/job_*/results.csv` into `outputs/all_results.csv`,
builds a VRT mosaic of the positive masks, and vectorises them to GeoJSON for
QGIS. Prints the tile / positive / download-failure counts — worth reading, and
required if you plan to re-segment (step 99 reads `all_results.csv`, and its
array size derives from the positive count).

Refuses to run if the number of `job_*/` directories doesn't match the
expected job count from `tile_grid.parquet` — a guard against silently merging
a run where an array task crashed. If that happens, check
`slurm_logs/infer/` for the failed task(s) and resubmit them individually
(`sbatch --array=<idx> slurm/01_inference_array.sh`) before merging again.

At the end it also prints `positives: <old> → <new> (Δ%)`, a best-effort
comparison of this run's positive count against the `inputs_index.csv`
currently published on HF, with a loud warning if it moved by more than 50% —
the only guard against a classifier that got fine-tuned into flagging almost
everything. Advisory only: it never blocks the merge, and if the previous
index can't be downloaded (offline, no prior export, no token) it just prints
a note and skips the comparison.

### 03 — Export to the frontend

Builds the final artefacts in `outputs_final/` and uploads them to the HF
dataset `alexKlg/windthrowAI-dataset-segmentation_results`:

- `windthrow.geojson` — dissolved windthrow polygons (map vector layer)
- `tiles/` — PNG tile pyramid, Web Mercator `z/x/y` (map raster overlay)
- `inputs/` — per-tile 7-band model input, uint16 web encoding (Expert-Review tab)
- `inputs_index.csv` — the tile list the Expert-Review tab reads: `tile_id` →
  bbox plus the per-tile scores from `results.csv` (`coverage`, `clf_prob`,
  `seg_mean_prob_in_mask`, `seg_max_prob`, `valid_frac`, `n_scenes`,
  `date_first`, `date_last`, `cloud_mean`, `source`, `scan_id`)
- `masks/` — per-tile proposed mask (Expert-Review tab)
- `probs/` — per-tile segmenter probability map, only with `--include-probs`
- `scan_meta.json` — this export's `scan_id`, timestamp, pinned model
  provenance and positive-tile count (repo root)

The masks come from `outputs/`, overridden per tile by `outputs_resegment/`
where step 99 has produced one. With no re-segmentation, the original masks are
exported unchanged. `source` in the index records which of the two a tile came
from; re-segmented tiles keep the `clf_prob` from the original scan, since
step 99 never re-runs the classifier.

Every export is stamped with a `scan_id` (`scan-YYYYMMDDTHHMMSSZ`, UTC of the
export — see AUTOMATION_PLAN.md contract 4.1, or override with `--scan-id`).
It lands as the last column of `inputs_index.csv`, in `scan_meta.json`, and —
once every upload above has succeeded — as a git tag on the dataset repo
(`revision=scan_id` then reproduces exactly the inputs/masks reviewers saw;
skip the tag with `--no-tag`). **Reviews submitted from that point on belong
to that round**: the frontend reads the tagged `scan_id` off the index and
attaches it to every review it submits (see T3 in AUTOMATION_PLAN.md).

Upload order matters and is not configurable: geojson → tiles → inputs →
masks → probs → `scan_meta.json` → `inputs_index.csv` *last*, because the
index is what makes a tile visible to the Expert-Review tab — it must not go
up before that tile's imagery and mask are already there.

The script runs the full export: every rescan refreshes the GeoJSON, the map
overlay (`tiles/`), the index, `inputs/` and `masks/`, so new positive tiles
get their review imagery and reviewers never 404 on `inputs/{tile}.tif`.
`tiles/` is mirrored, not just added to: remote tiles the new scan no longer
produces are deleted in the same commit, so a detection the new model dropped
disappears from the map too. gdal2tiles over z5–13 is the slow part, hence the
6 h walltime; pass `--skip-tiles` by hand for a quick run that leaves the map
as it is.

`inputs_index.csv` is *not* tied to `--skip-inputs`: it is one small file, it
carries every per-tile score the review UI filters on, and skipping the bulk
input rasters is no reason to leave it stale. Use `--skip-index` to suppress it
explicitly.

`--dry-run` runs every export step and writes the local files, but skips the
HF upload and the tag, printing the upload plan instead — the only mode safe
to run without write access, and the one an agent verifying this pipeline
should use.

Needs `gdalbuildvrt` + `gdal2tiles.py` on PATH, and HF write access:
`huggingface-cli login` on the cluster, or `HF_TOKEN` exported before submitting.

## Exceptions

### 99 — Re-segment (array job, optional)

Re-runs *segmentation only* with a different checkpoint over the tiles already
flagged positive, skipping classification and re-downloading only that small
subset. This is how a newly validated segmenter reaches production without a
full re-scan. Writes `outputs_resegment/job_*/`, mirroring step 01's layout.

Run it between 02 and 03. The export overrides per tile, so a partial run is
fine — tiles it never touched keep their original mask, and a tile the new
segmenter rejects is dropped from the export.

Its array size comes from the positive count printed by step 02:

```bash
--array=0-$(( (num_positives + 499) / 500 - 1 ))   # with --chunk-size 500
```

### 99 — Recompress the published inputs (one-time)

Migrates the `inputs/` already on HF from raw float32/LZW (~2.0 MB per tile) to
the compact uint16 encoding (~0.6 MB), which is what made every patch in the
Expert-Review tab a multi-second download. Downloads `inputs/`, re-encodes,
uploads back.

One-off: step 03 writes that encoding directly now, and the script skips tiles
that are already uint16, so a re-run is a no-op. Dry run by default:

```bash
sbatch slurm/99_recompress_hf_inputs_one_time.sh            # report the size win only
sbatch slurm/99_recompress_hf_inputs_one_time.sh --upload   # publish
```

### 99 — Summarise the expert reviews

Read-only report on how much review signal has accumulated in the dataset,
split into what it can train. Run it before opening a retraining round — it is
what tells you whether there is enough, and how much reviewer redundancy sits
behind each tile.

```bash
sbatch slurm/99_summarize_reviews.sh                     # full report
sbatch slurm/99_summarize_reviews.sh --fast              # counts only, no JSON download
sbatch slurm/99_summarize_reviews.sh --since 2026-08-01  # only recent reviews
```

Reviews live in the same dataset as `reviews/{tile_id}/{ts}__p{pixels}__{reviewer}.json`
plus the corrected mask PNG next to it, written by the tileloading backend. One
record carries both training signals:

| Signal | Comes from | Reported as |
|--------|-----------|-------------|
| Classification | the verdict — `confirmed`/`edited` is a confirmed true positive, `no_windthrow` a corrected false positive | labelled tiles, labels per tile, majority label per tile, unanimous vs. split among multi-reviewed tiles |
| Segmentation | the mask PNG — `edited` is a redrawn mask, `confirmed` endorses the proposed one, `no_windthrow` implies an empty one | distinct corrected tiles, usable mask targets per tile, pixels changed |

Every reviewed tile passed the stage-1 gate, so the classification labels are
confirmations and false-positive corrections of the *current* classifier — the
round adds no new negatives from outside the positive set.

`--fast` works off the file listing alone (it encodes timestamp, changed-pixel
count and reviewer). That covers the counts and the per-tile averages but not
the verdicts, so the classification/segmentation split needs a full run.

Two CSVs land in `outputs_final/reviews/` (`--no-csv` to suppress):
`review_summary_per_tile.csv` — one row per reviewed tile with its verdict
counts, majority label, whether the reviewers agreed and the latest mask path,
which is the label set a retraining reads — and `review_summary_per_reviewer.csv`.

## Full run

```bash
cd windthrowAI-amazon-scan
slurm/run_full_scan.sh
```

This submits `00` and waits for it (~15 min — run it under `tmux`/`screen` or
`nohup ... & disown` since an SSH drop during the wait aborts the whole chain
before `01` is ever submitted), reads the resulting array size from
`tile_grid.parquet`, then chains `01 → 02 → 03` via
`sbatch --dependency=afterok:<jobid>` and returns immediately — SLURM carries
the rest even if you log out. It prints every job id for `squeue -u $USER`.
Starting a scan is still a deliberate, manual action; nothing triggers this
automatically, including a new model being published in `windthrowAI-training`.

`02` and `03` have `--kill-on-invalid-dep=yes` set, so if any `01` array task
fails, the dependent jobs show up as `CANCELLED` in `squeue` instead of sitting
silently `PENDING (DependencyNeverSatisfied)` forever.

The optional re-segment step (99) is **not** part of this chain — it stays a
manual, reviewed insertion between 02 and 03:

```bash
# after 02, read the positive count from slurm_logs/merge/*.out
sbatch --array=0-120 slurm/99_resegment_array_optional.sh
sbatch --dependency=afterok:<reseg_jobid> slurm/03_export_frontend.sh
```

### Running the steps by hand

Equivalent to `run_full_scan.sh`, spelled out — useful for debugging or when
you want to inspect step 00's output before committing to the array size:

```bash
cd windthrowAI-amazon-scan

sbatch slurm/00_generate_grid.sh
# read the printed "--array=0-N" from slurm_logs/grid/*.out

sbatch --array=0-115 slurm/01_inference_array.sh
sbatch --dependency=afterok:<infer_jobid> slurm/02_merge_results.sh
sbatch --dependency=afterok:<merge_jobid> slurm/03_export_frontend.sh
```

(To swap in a new segmenter here, insert the optional step before the export —
see the example under "Full run" above.)

`--array` on the command line overrides the `#SBATCH --array` in the file, so
the ranges can be set at submit time instead of editing the scripts.

Both array jobs are per-task independent — a failed task can be resubmitted on
its own with `--array=<idx>`, and completed tasks are safe to re-run (each
rewrites its own `job_{idx:04d}/` directory).

Logs land in `slurm_logs/`: `-%j` for single jobs, `-%A_%a` for array tasks.

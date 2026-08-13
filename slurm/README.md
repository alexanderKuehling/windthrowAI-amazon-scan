# SLURM pipeline

Amazon-wide windthrow scan, from an empty repo to the Hugging Face dataset the
frontends read. Submit every job **from the repo root** — the scripts use paths
relative to it (`scripts/`, `outputs/`, `slurm_logs/`):

```bash
cd windthrowAI-amazon-scan
sbatch slurm/00_generate_grid.sh
```

All jobs activate the `windthrowEnv` conda environment. The GEE-facing and
HF-facing jobs load `ml proxy` for outbound network access.

`00`–`03` are the standard pipeline and run in that order. `99_*` scripts are
exceptions: neither is part of a normal scan.

To re-run a scan from scratch, clear the previous one first —
[`reset.md`](../reset.md) lists what to delete and what to keep.

| # | Script | Job | GPU | Array | Walltime | Reads | Writes |
|---|--------|-----|-----|-------|----------|-------|--------|
| 00 | `00_generate_grid.sh` | `WTAI_grid` | no | – | 15 min | `src/config.py` | `tile_grid.parquet` |
| 01 | `01_inference_array.sh` | `WTAI_infer` | 2×A100 | yes | 4 h | `tile_grid.parquet` | `outputs/job_*/` |
| 02 | `02_merge_results.sh` | `WTAI_merge` | no | – | 1 h | `outputs/job_*/results.csv` | `outputs/all_results.csv`, `.vrt`, `.geojson` |
| 03 | `03_export_frontend.sh` | `WTAI_export` | no | – | 2 h | `outputs/`, `outputs_resegment/` | `outputs_final/` + HF dataset |
| 99 | `99_resegment_array_optional.sh` | `WTAI_reseg` | 2×A100 | yes | 2 h | `outputs/all_results.csv` | `outputs_resegment/job_*/` |
| 99 | `99_recompress_hf_inputs_one_time.sh` | `WTAI_recompress` | no | – | 3 h | HF dataset `inputs/` | HF dataset `inputs/` |

## The standard pipeline

### 00 — Generate the tile grid

Builds the spatial grid covering the Amazon (256 px tiles at 30 m, Landsat-8
native resolution) and writes `tile_grid.parquet`. Run once; re-run only after
changing the grid parameters in `src/config.py`.

**Its output tells you the array size for step 01** — it prints the exact flag:

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

Set `--array` to the range printed by step 00. The checkpoint and its encoder
must match; `--segmenter-path`/`--backbone` in the script override the defaults
in `src/config.py` (currently `segmenter.pth` / `vgg19_bn`). Dates default to
the dry season, `2024-06-01`–`2024-09-30`.

Needs GEE credentials: `gee-windthrowai-secret.json` in the repo root
(or `--key-path`).

### 02 — Merge results

Concatenates every `outputs/job_*/results.csv` into `outputs/all_results.csv`,
builds a VRT mosaic of the positive masks, and vectorises them to GeoJSON for
QGIS. Prints the tile / positive / download-failure counts — worth reading, and
required if you plan to re-segment (step 99 reads `all_results.csv`, and its
array size derives from the positive count).

### 03 — Export to the frontend

Builds the final artefacts in `outputs_final/` and uploads them to the HF
dataset `alexKlg/windthrowAI-dataset-segmentation_results`:

- `windthrow.geojson` — dissolved windthrow polygons (map vector layer)
- `tiles/` — PNG tile pyramid, Web Mercator `z/x/y` (map raster overlay)
- `inputs/` — per-tile 7-band model input, uint16 web encoding (Expert-Review tab)
- `inputs_index.csv` — the tile list the Expert-Review tab reads: `tile_id` →
  bbox plus the per-tile scores from `results.csv` (`coverage`, `clf_prob`,
  `seg_mean_prob_in_mask`, `seg_max_prob`, `valid_frac`, `n_scenes`,
  `date_first`, `date_last`, `cloud_mean`, `source`)
- `masks/` — per-tile proposed mask (Expert-Review tab)
- `probs/` — per-tile segmenter probability map, only with `--include-probs`

The masks come from `outputs/`, overridden per tile by `outputs_resegment/`
where step 99 has produced one. With no re-segmentation, the original masks are
exported unchanged. `source` in the index records which of the two a tile came
from; re-segmented tiles keep the `clf_prob` from the original scan, since
step 99 never re-runs the classifier.

The script currently passes `--skip-tiles --skip-inputs`, i.e. it refreshes only
the GeoJSON, the index and the masks. **Drop those flags to publish a full
result set.** `--skip-tiles` is the expensive one (gdal2tiles over z5–13).

`inputs_index.csv` is *not* tied to `--skip-inputs`: it is one small file, it
carries every per-tile score the review UI filters on, and skipping the bulk
input rasters is no reason to leave it stale. Use `--skip-index` to suppress it
explicitly.

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

## Full run

Step 01 needs an `--array` range that is only known once step 00 has printed it,
so the pipeline is not a single dependency chain end to end.

```bash
cd windthrowAI-amazon-scan

sbatch slurm/00_generate_grid.sh
# read the printed "--array=0-N" from slurm_logs/grid-*.out

sbatch --array=0-115 slurm/01_inference_array.sh
sbatch --dependency=afterok:<infer_jobid> slurm/02_merge_results.sh
sbatch --dependency=afterok:<merge_jobid> slurm/03_export_frontend.sh
```

To swap in a new segmenter, insert the optional step before the export:

```bash
# after 02, read the positive count from slurm_logs/merge-*.out
sbatch --array=0-120 slurm/99_resegment_array_optional.sh
sbatch --dependency=afterok:<reseg_jobid> slurm/03_export_frontend.sh
```

`--array` on the command line overrides the `#SBATCH --array` in the file, so
the ranges can be set at submit time instead of editing the scripts.

Both array jobs are per-task independent — a failed task can be resubmitted on
its own with `--array=<idx>`, and completed tasks are safe to re-run (each
rewrites its own `job_{idx:04d}/` directory).

Logs land in `slurm_logs/`: `-%j` for single jobs, `-%A_%a` for array tasks.

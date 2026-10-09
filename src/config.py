import os
import numpy as np
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).parent.parent

# ── Amazon bounding box  (lon_min, lat_min, lon_max, lat_max) ─────────────
AMAZON_BBOX: Tuple[float, float, float, float] = (-85.0, -15.0, -45.0, 6.0)

# ── Landsat-8 bands  (order = model input order: R,G,B,NIR,SWIR1,SWIR2) ──
BAND_NAMES      = ["Red", "Green", "Blue", "nIR", "swIR1", "swIR2", "NBR"]
NUM_MODEL_BANDS = 7   # 6 spectral + NBR computed on-the-fly

# ── Tiling ────────────────────────────────────────────────────────────────
TILE_SIZE_PX    = 256
RESOLUTION_M    = 30      # Landsat-8 native resolution
OVERLAP_PX      = 32

# ── Scene search ──────────────────────────────────────────────────────────
DATE_FROM       = "2024-06-01"
DATE_TO         = "2024-09-30"   # Amazon dry season → lowest cloud cover
CLOUD_COVER_MAX = 70

# ── NBR band encoding ─────────────────────────────────────────────────────
# "raw": NBR = (NIR - SWIR2) / (NIR + SWIR2) in -1..1 — what both live models
#        were trained on (band 8 of ultimate_dataset, NORM_P5/P95 below), what
#        the segmentation frontend and both gradio Spaces compute.
# "rescaled": (NBR + 1) / 2 in 0..1 — what this scan used to feed the models.
#        Forest NBR (~0.73 raw) then sits at ~0.87, above NORM_P95, so the
#        channel saturates: on the original training set's holdout the live
#        segmenter's Dice falls from 0.535 to 0.350 under it (classifier F1
#        0.983 -> 0.980). Every round published before 2026-10 is "rescaled".
# The encoding is recorded per tile (run_meta.json -> inputs_index.csv
# `nbr_encoding`), so the review app and windthrowAI-training read either.
NBR_ENCODING = "raw"

# ── Normalization – fixed values from the training dataset ─────────────────
# Source: best_params_classify_optimized.json  (windthrowAI-classification-gradio)
# Do NOT recompute on Amazon data – use exactly these training-time values.
NORM_P5  = np.array([0.01835, 0.0295425, 0.01351, 0.21415,
                      0.102775, 0.0347125, 0.48246968], dtype=np.float32)
NORM_P95 = np.array([0.063285, 0.0722225, 0.0521475, 0.355225,
                      0.191545, 0.0954875, 0.7648977],  dtype=np.float32)

# ── Models ────────────────────────────────────────────────────────────────
# Backbones are fallbacks only — load_classifier/load_segmenter read the
# checkpoint's own "backbone" field first. They match what is live: the
# classifier embeds efficientnet_b0; the segmenter repo below holds a bare
# smp.Unet(resnet34) state_dict with no backbone embedded, so this fallback is
# what makes the pinned HF pull in step 01 load it.
CLASSIFIER_BACKBONE  = "efficientnet_b0"
CLASSIFIER_DROPOUT   = 0.2
CLASSIFIER_THRESHOLD = 0.5

SEGMENTER_BACKBONE  = "resnet34"
SEGMENTER_THRESHOLD = 0.5

# ── Paths ─────────────────────────────────────────────────────────────────
MODELS_DIR   = ROOT / "models"
OUTPUTS_DIR  = ROOT / "outputs"
GRID_FILE    = ROOT / "tile_grid.parquet"
DATASET_PATH = ROOT.parent / "ultimate_dataset"   # ../ultimate_dataset from repo root
GEE_KEY_PATH = ROOT / "gee-windthrowai-secret.json"

CLASSIFIER_PTH = MODELS_DIR / "classifier.pth"
SEGMENTER_PTH  = MODELS_DIR / "segmenter.pth"

# ── HF Hub model registry (mirrors windthrowAI-training/src/config.py) ─────
# Duplicated rather than imported — these constants change far less often than
# the pipeline code around them. Kept in sync by hand with the training repo
# and with windthrowAI-classification-gradio / windthrowAI-segmentation-gradio.
CLASSIFIER_REPO     = "alexKlg/windthrowAI-classifier"
CLASSIFIER_FILENAME = "best_classifier_optimized.pth"
SEGMENTER_REPO      = "alexKlg/windthrow-unet-classifier-pretrained"   # decision D1
SEGMENTER_FILENAME  = "best_unet_classifier_pretrained_seg.pth"

# Written once per scan by scripts/resolve_model_versions.py, pinning the exact
# HF commit revision every array task in slurm/01_inference_array.sh must use.
MODEL_VERSIONS_FILE = ROOT / "model_versions.json"

# ── Inference ─────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
TILES_PER_JOB = 2000
# An array task downloads, classifies and writes its tiles this many at a time,
# so a task that hits its walltime keeps everything but the last chunk and its
# resubmit downloads only what is missing. 500 = two batched scene-metadata
# requests on GEE (GEE_STATS_CHUNK) and four full 2-GPU batches.
DOWNLOAD_CHUNK = 500

# ── Merge checks (scripts/merge_results.py) ───────────────────────────────
# Step 02 exits non-zero — which cancels the chained export (03), so reviewers
# never see the round — when tiles are missing from the results, when more than
# this share of tiles with imagery failed to download, or when the positive
# count moved by more than MERGE_MAX_POSITIVE_DELTA against the published index
# (a drifted classifier, or a broken download). Override deliberately with
# --allow-errors / --accept-delta; slurm/resume_scan.sh finishes the scan.
MERGE_MAX_ERROR_FRACTION = 0.01
MERGE_MAX_POSITIVE_DELTA = 0.50

# ── Imagery source ────────────────────────────────────────────────────────
# "pc":  Microsoft Planetary Computer (src/data_pc.py), the default since
#        2026-10-05 — the same USGS Collection 2 Level-2 scenes as public files,
#        composited here with GEE's recipe, no compute quota. 19 of 20 live
#        tiles were 99.4-99.98% bit-identical to GEE; PC lacks ~5% of GEE's
#        scenes, and a tile whose best scene is among them gets the next best.
# "gee": Google Earth Engine (src/data.py) — metered per project (monthly EECU
#        tiers since 2026-04-27; the cause of the scan's rate limits).
# Override per run with WINDTHROW_DATA_SOURCE (SLURM passes it to every task)
# or run_inference.py --source; scripts/compare_sources.py compares the two.
DATA_SOURCE = os.environ.get("WINDTHROW_DATA_SOURCE", "pc")

# ── Export rule: which flagged tiles count as detections ──────────────────
# The classifier flags ~60% of scan tiles at CLASSIFIER_THRESHOLD, and on most
# of them the segmenter draws nothing or a few stray pixels (80 random tiles,
# 2026-10-05: 60% flagged, half with no mask pixel at all). Only tiles that are
# flagged *and* carry a windthrow-sized patch are exported — inputs_index.csv
# (the review app), masks/, inputs/, the map overlay and the GeoJSON; on that
# sample ~10% of all tiles. The patch is the largest 8-connected mask
# component (results.csv `max_patch_px`): 30 px = 2.7 ha at 30 m, and 95% of
# ultimate_dataset's labelled windthrow tiles have a patch at least that big —
# a coverage share would not do, since 1% of a tile (655 px) keeps only 25%
# of them. Applied at merge/export time from the recorded scores, so changing
# these needs no rescan: re-run 02 -> 03. Flagged tiles below the rule join
# the spot-check pool like rejected ones (src/results.py::is_detection).
EXPORT_MIN_CLF_PROB = CLASSIFIER_THRESHOLD
EXPORT_MIN_PATCH_PX = int(os.environ.get("WINDTHROW_EXPORT_MIN_PATCH_PX", 30))   # env: per run, no file edit

# Hugging Face rejects a commit that leaves more than this many files in one
# folder. The export therefore files each tile under its grid row
# (inputs/r0341/r0341_c0444.tif, see export_for_frontend.tile_file): a few
# hundred tiles per row, a few hundred rows, however big a round gets.
HF_MAX_FILES_PER_DIR = 10_000

# ── Spot-check sample ("control" tiles) ───────────────────────────────────
# Reviewers otherwise only ever see tiles the scan reported, so its misses and
# any drift toward flagging plain forest stay invisible. A fixed random share
# of the tiles it does *not* report — rejected by the classifier, or flagged
# without a patch passing the export rule above — is exported with an empty
# mask and mixed into review (see src/results.py::is_control_sample). Only
# tiles clear enough to judge qualify. ~0.5% of them is several hundred per
# Amazon scan — plenty, as the review app shows one about every 10th tile.
CONTROL_SAMPLE_RATE    = 0.005
CONTROL_MIN_VALID_FRAC = 0.8

# ── GEE ───────────────────────────────────────────────────────────────────
GEE_PROJECT = "gee-windthrowai"

# Request budget. The project allows 40 concurrent requests (HTTP 429 beyond
# that), shared with the tileloading backend on Cloud Run. The scan's share is
# GEE_ARRAY_PARALLEL × GEE_WORKERS = 32, leaving headroom for the live app.
# GEE_ARRAY_PARALLEL is enforced by SLURM, not Python: it is the %N in
# `sbatch --array=0-115%8` — submitting without it removes the cap.
GEE_WORKERS        = 4     # concurrent pixel downloads per array task
GEE_ARRAY_PARALLEL = 8     # array tasks running at once
GEE_STATS_CHUNK    = 250   # tiles per batched scene-metadata request
GEE_RETRIES        = 4     # attempts per request, on top of the ee client's own 429 backoff
GEE_HIGH_VOLUME    = True  # Google's endpoint for many parallel automated requests

# ── Microsoft Planetary Computer (DATA_SOURCE = "pc") ─────────────────────
# Reads are plain HTTPS range requests against Azure Blob storage, signed with
# a short-lived SAS token the planetary_computer package fetches and caches.
# No compute quota, so the only budget is bandwidth: PC_WORKERS tiles in
# flight per task, PC_ARRAY_PARALLEL tasks at once. An optional free API key
# (env PC_SDK_SUBSCRIPTION_KEY) raises the token endpoint's rate limit.
PC_STAC_URL       = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_COLLECTION     = "landsat-c2-l2"
PC_WORKERS        = 16
PC_ARRAY_PARALLEL = 16
PC_RETRIES        = 4

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

# ── Spot-check sample ("control" tiles) ───────────────────────────────────
# Reviewers otherwise only ever see tiles the classifier flagged, so its misses
# and any drift toward flagging plain forest stay invisible. A fixed random
# share of the tiles it *rejects* is exported with an empty mask and mixed into
# review (see src/results.py::is_control_sample). Only tiles clear enough to
# judge qualify. ~0.5% of rejected tiles is a few hundred per Amazon scan —
# plenty, as the review app shows one about every 10th tile.
CONTROL_SAMPLE_RATE    = 0.005
CONTROL_MIN_VALID_FRAC = 0.8

# ── GEE ───────────────────────────────────────────────────────────────────
GEE_PROJECT = "gee-windthrowai"

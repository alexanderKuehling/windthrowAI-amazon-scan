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
TILE_SIZE_PX    = 1024
RESOLUTION_M    = 30      # Landsat-8 native resolution
OVERLAP_PX      = 64

# ── Scene search ──────────────────────────────────────────────────────────
DATE_FROM       = "2024-06-01"
DATE_TO         = "2024-09-30"   # Amazon dry season → lowest cloud cover
CLOUD_COVER_MAX = 30

# ── Normalization – fixed values from the training dataset ─────────────────
# Source: best_params_classify_optimized.json  (windthrowAI-classification-gradio)
# Do NOT recompute on Amazon data – use exactly these training-time values.
NORM_P5  = np.array([0.01835, 0.0295425, 0.01351, 0.21415,
                      0.102775, 0.0347125, 0.48246968], dtype=np.float32)
NORM_P95 = np.array([0.063285, 0.0722225, 0.0521475, 0.355225,
                      0.191545, 0.0954875, 0.7648977],  dtype=np.float32)

# ── Models ────────────────────────────────────────────────────────────────
CLASSIFIER_BACKBONE  = "mobilenetv3_small_100"
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

# ── Inference ─────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
TILES_PER_JOB = 500

# ── GEE ───────────────────────────────────────────────────────────────────
GEE_PROJECT = "gee-windthrowai"

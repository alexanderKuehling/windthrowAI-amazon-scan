"""Per-tile result records shared by the inference and re-segmentation passes.

Both write the same columns so the export reads outputs/ and outputs_resegment/
through one code path. Everything after `download_ok` is scan-derived evidence
that cannot be recovered without re-downloading the imagery and re-running the
models — it is written even for tiles the run rejects.
"""

import csv
import hashlib
from pathlib import Path
from typing import Optional, Tuple

from src.config import (
    CONTROL_MIN_VALID_FRAC,
    CONTROL_SAMPLE_RATE,
    EXPORT_MIN_CLF_PROB,
    EXPORT_MIN_PATCH_PX,
)

CSV_COLUMNS = [
    "tile_id", "lon_min", "lat_min", "lon_max", "lat_max",
    "is_positive", "mask_path", "input_path", "download_ok",
    "clf_prob",                 # stage-1 windthrow probability (every downloaded tile)
    "valid_frac",               # share of non-cloud-masked pixels
    "n_scenes", "date_first", "date_last", "cloud_mean",   # composite provenance
    "coverage", "mask_pixels",  # extent of the proposed mask
    "max_patch_px",             # largest connected mask patch — the export rule's measure
    "seg_mean_prob", "seg_mean_prob_in_mask", "seg_max_prob",  # stage-2 confidence
    "prob_path",                # segmenter probability map (uint8, DN/255)
    "is_control",               # rejected tile exported as a review spot check
]


def largest_patch_px(mask) -> int:
    """Pixel count of the largest 8-connected windthrow patch in a 0/1 mask."""
    import numpy as np
    import rasterio.features
    from shapely.geometry import shape

    m = np.asarray(mask).astype("uint8")
    if not m.any():
        return 0
    return int(max(shape(g).area for g, _ in rasterio.features.shapes(m, mask=m.astype(bool), connectivity=8)))


def patch_px_of(row: dict, root: Path) -> int:
    """A results row's max_patch_px, read from its mask file when the run that
    wrote it predates the column (0 when there is no mask)."""
    raw = row.get("max_patch_px")
    try:
        if raw not in (None, "") and str(raw).lower() != "nan":
            return int(float(raw))
    except (TypeError, ValueError):
        pass
    mask_path = str(row.get("mask_path") or "").strip()
    if not mask_path or mask_path.lower() == "nan" or not (root / mask_path).exists():
        return 0
    import rasterio
    with rasterio.open(root / mask_path) as src:
        return largest_patch_px(src.read(1) > 0)


def is_detection(clf_prob, max_patch_px: int) -> bool:
    """The export rule (config.EXPORT_MIN_CLF_PROB / EXPORT_MIN_PATCH_PX): a
    flagged tile counts as a detection — exported, reviewed, on the map — only
    with a windthrow-sized patch. Shared by merge_results.py (the positive count
    it guards) and export_for_frontend.py (what it publishes). A row from before
    clf_prob was recorded passed the classifier gate at the time."""
    try:
        if clf_prob not in (None, "") and float(clf_prob) < EXPORT_MIN_CLF_PROB:
            return False
    except (TypeError, ValueError):
        pass
    return max_patch_px >= EXPORT_MIN_PATCH_PX


def is_control_sample(tile_id: str, valid_frac) -> bool:
    """Whether a tile the scan does not report (rejected by the classifier, or
    flagged without passing the export rule) is one of the review spot checks.

    Chosen by a hash of the tile id, so it is deterministic (the same tiles in
    every scan, and a resumed job agrees with itself) and independent of the
    model. The "control:" salt keeps it uncorrelated with windthrowAI-training's
    train/val/test split, which hashes the bare tile id: unsalted, every control
    tile would land in the test split. Tiles mostly hidden by cloud are left out,
    since a reviewer couldn't judge them."""
    try:
        if float(valid_frac) < CONTROL_MIN_VALID_FRAC:
            return False
    except (TypeError, ValueError):
        return False
    digest = hashlib.md5(f"control:{tile_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < CONTROL_SAMPLE_RATE


def is_retryable(row: dict) -> bool:
    """Whether a results row is a failed download worth another attempt.

    Every failure except "GEE holds no imagery here" (n_scenes == 0: open ocean,
    nothing under the cloud limit): quota rejections, timeouts and dropped
    connections leave n_scenes non-zero or empty. Works on csv.DictReader rows
    (strings) and on fresh base_row() dicts alike."""
    return str(row.get("download_ok")) == "False" and str(row.get("n_scenes")) != "0"


def open_results_csv(path: Path) -> Tuple:
    """Open a per-job results CSV for append → (file, DictWriter, done_ids).

    A CSV left by an earlier run has fewer columns; it is rewritten in place with
    the current header (old rows padded with empty fields) so a resumed job keeps
    its finished tiles instead of appending rows that no longer line up. Rows of
    retryable failures (see is_retryable) are dropped in the same rewrite, so a
    resubmitted job downloads exactly those tiles again."""
    done_ids: set = set()
    if path.exists():
        with open(path, newline="") as f:
            reader     = csv.DictReader(f)
            old_header = reader.fieldnames or []
            rows       = list(reader)
        n_retry  = sum(1 for r in rows if is_retryable(r))
        rows     = [r for r in rows if not is_retryable(r)]
        done_ids = {r["tile_id"] for r in rows if r.get("tile_id")}
        if n_retry:
            print(f"  retrying {n_retry} tiles whose download failed last time")
        if old_header != CSV_COLUMNS or n_retry:
            if old_header != CSV_COLUMNS:
                print(f"  migrating {path.name}: {len(old_header)} -> {len(CSV_COLUMNS)} columns")
            with open(path, "w", newline="\n") as f:
                w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({c: (r.get(c) or "") for c in CSV_COLUMNS})

    has_header = path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    fh     = open(path, "a", newline="\n")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if not has_header:
        writer.writeheader()
    return fh, writer, done_ids


def base_row(
    tile_id:     str,
    bbox:        Tuple[float, float, float, float],
    meta:        Optional[dict] = None,
    download_ok: bool = True,
) -> dict:
    """Row skeleton shared by failed downloads, negatives and positives.

    *meta* is the composite provenance from download_tiles (either source); its fields
    stay empty when the download never produced imagery."""
    meta = meta or {}
    lon_min, lat_min, lon_max, lat_max = bbox
    return {
        "tile_id":     tile_id,
        "lon_min":     lon_min,
        "lat_min":     lat_min,
        "lon_max":     lon_max,
        "lat_max":     lat_max,
        "is_positive": False,
        "mask_path":   "",
        "input_path":  "",
        "download_ok": download_ok,
        "clf_prob":    "",
        "valid_frac":  meta.get("valid_frac", ""),
        "n_scenes":    meta.get("n_scenes", ""),
        "date_first":  meta.get("date_first", ""),
        "date_last":   meta.get("date_last", ""),
        "cloud_mean":  meta.get("cloud_mean", ""),
        "coverage":    "",
        "mask_pixels": "",
        "max_patch_px": "",
        "seg_mean_prob": "",
        "seg_mean_prob_in_mask": "",
        "seg_max_prob":  "",
        "prob_path":     "",
        "is_control":    False,
    }


def fill_segmentation(record: dict, prob, mask) -> dict:
    """Add the stage-2 extent + confidence columns for a segmented tile.

    prob : float (H, W) per-pixel probability; mask : uint8 (H, W) binarisation.
    """
    on = mask.astype(bool)
    record["mask_pixels"]           = int(on.sum())
    record["coverage"]              = round(float(on.mean()) * 100.0, 4)
    record["seg_mean_prob"]         = round(float(prob.mean()), 6)
    record["seg_max_prob"]          = round(float(prob.max()), 6)
    record["seg_mean_prob_in_mask"] = round(float(prob[on].mean()), 6) if on.any() else 0.0
    record["max_patch_px"]          = largest_patch_px(mask)
    return record

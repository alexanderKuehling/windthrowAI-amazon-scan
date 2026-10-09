"""
Build the final windthrow result set and push it to the Hugging Face dataset
used by the frontend, as both a vector layer (GeoJSON) and a raster tile
pyramid (PNG, Web Mercator z/x/y) for map overlay.

    python scripts/export_for_frontend.py

Steps:
  1. Pick the final mask per tile: outputs/ (original run), overridden per tile by
     outputs_resegment/ (new segmenter) where it exists. Re-segmentation is optional.
     Then apply the export rule (config.EXPORT_MIN_CLF_PROB / EXPORT_MIN_PATCH_PX,
     src/results.py::is_detection): only flagged tiles with a windthrow-sized mask
     patch are detections. Steps 2-6 publish those (plus the spot checks, below).
  2. Vectorize all final masks into one dissolved GeoJSON.
  3. Convert each mask to a red/transparent RGBA GeoTIFF, build a VRT mosaic,
     and render a PNG tile pyramid with gdal2tiles.py.
  4. Stage each positive tile's 7-band model input as a compact uint16 GeoTIFF
     (encode_input_for_web).
  5. Write inputs_index.csv: tile_id -> bbox plus the per-tile scores the scan
     recorded (windthrow `coverage` %, classifier confidence `clf_prob`, segmenter
     confidence, `valid_frac` / `cloud_mean` / acquisition dates), so the
     Expert-Review tab can filter and prioritise tiles without fetching imagery.
     The last column is `scan_id` (see below).
  6. Stage each positive tile's proposed mask as a single-band uint8 GeoTIFF
     (masks/{tile_id}.tif), aligned to inputs/{tile_id}.tif, for the Expert-Review tab.
  7. Optionally (--include-probs) stage the segmenter probability maps as
     probs/{tile_id}.tif (uint8, probability = DN/255).
  8. Write scan_meta.json (model provenance + tile counts for this export).

Steps 4-6 also cover the spot-check tiles: a hash sample of the tiles the scan
does not report — rejected tiles run_inference kept an input for (is_control),
and flagged tiles below the export rule (same src/results.py::is_control_sample).
They get an all-zero mask and `kind=control` in the index, and stay out of the
GeoJSON and the map (steps 2-3), which show detections only. Every other index
row is a detection, `kind=flagged`.
  9. Upload, in an order a reviewer can safely browse mid-upload: geojson ->
     tiles -> inputs -> masks -> probs -> scan_meta.json -> inputs_index.csv
     *last* (AUTOMATION_PLAN.md contract 4.2/4.3, ticket T2/T8). The index is
     what makes a tile visible to the Expert-Review tab, so it must not go up
     before that tile's imagery and mask are already there. On success, tag
     the dataset repo with scan_id (skip with --no-tag).

Every export is stamped with a `scan_id` (format `scan-YYYYMMDDTHHMMSSZ`,
UTC of the export; see AUTOMATION_PLAN.md contract 4.1). It is computed
automatically, or pinned with `--scan-id` (e.g. to reproduce `scan-legacy`).
`--dry-run` runs every step above except the actual HF upload and the tag,
and prints what would be uploaded instead — the only mode a reviewing agent
should ever run.

Before uploading, the export checks that the dataset still has what the
reviews from before scan_id existed point at: the one-time `scan-legacy` tag
(LOOP.md). It refuses to run the first scan_id export without it, since that
export overwrites inputs/ and masks/ (`--allow-untagged-legacy` overrides).
`--check-legacy-tag` runs only that check; slurm/run_full_scan.sh calls it
before submitting anything.

Requires on PATH for the map overlay (step 3): gdalbuildvrt and gdal2tiles.py,
GDAL's command-line tools — checked before anything else runs; slurm/03 finds
them in a separate conda env `gdal` if windthrowEnv lacks them. --skip-tiles
publishes without refreshing the map.
Requires Python packages: rasterio, geopandas, shapely, huggingface_hub.
Requires `huggingface-cli login` (or HF_TOKEN env var) with write access to
the target dataset repo.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import EXPORT_MIN_CLF_PROB, EXPORT_MIN_PATCH_PX, HF_MAX_FILES_PER_DIR, MODEL_VERSIONS_FILE
from src.results import is_control_sample, is_detection, passes_classifier, patch_px_of

RESEG_DIR    = ROOT / "outputs_resegment"
ORIG_DIR     = ROOT / "outputs"
WORK_DIR     = ROOT / "outputs_final"
RGBA_DIR     = WORK_DIR / "rgba_masks"
TILES_DIR    = WORK_DIR / "tiles"
INPUTS_DIR   = WORK_DIR / "inputs"
MASKS_DIR    = WORK_DIR / "masks"
PROBS_DIR    = WORK_DIR / "probs"
VRT_PATH     = WORK_DIR / "mosaic.vrt"
GEOJSON_PATH = WORK_DIR / "windthrow.geojson"
INPUTS_INDEX_PATH = WORK_DIR / "inputs_index.csv"
SCAN_META_PATH    = WORK_DIR / "scan_meta.json"

HF_REPO_ID   = "alexKlg/windthrowAI-dataset-segmentation_results"

# AUTOMATION_PLAN.md contract 4.1 — used by every scan_id validator across the
# four repos (tileloading, expert-review, training, amazon-scan).
SCAN_ID_RE = re.compile(r"^scan-(legacy|\d{8}T\d{6}Z)$")
LEGACY_SCAN_ID = "scan-legacy"

# Web encoding of the model inputs: reflectance = DN * WEB_SCALE + WEB_OFFSET.
# uint16 + deflate is ~3.5x smaller than the raw float32/LZW tif (see encode_input_for_web).
WEB_SCALE  = 1e-4
WEB_OFFSET = -1.0   # DN 0..65535 -> -1.0 .. 5.5535 (covers negative NBR)


def default_scan_id() -> str:
    return "scan-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=HF_REPO_ID)
    ap.add_argument("--zoom",    default="5-13", help="gdal2tiles.py -z range")
    ap.add_argument("--scan-id", default=None,
                    help="override the auto-generated scan_id (default: "
                         "scan-<UTC now>). Must match "
                         f"{SCAN_ID_RE.pattern!r} (AUTOMATION_PLAN.md contract 4.1), "
                         "e.g. to reproduce scan-legacy.")
    ap.add_argument("--skip-tiles", action="store_true", help="only export GeoJSON, skip raster tiles")
    ap.add_argument("--skip-inputs", action="store_true", help="skip exporting model-input GeoTIFFs")
    ap.add_argument("--skip-masks", action="store_true", help="skip exporting per-tile mask GeoTIFFs")
    ap.add_argument("--skip-index", action="store_true",
                    help="skip inputs_index.csv (the tile list + per-tile scores the "
                         "Expert-Review tab reads); it is cheap and is written even "
                         "with --skip-inputs")
    ap.add_argument("--include-probs", action="store_true",
                    help="also publish probs/{tile_id}.tif, the segmenter probability maps")
    ap.add_argument("--no-tag", action="store_true",
                    help="skip tagging the HF dataset repo with scan_id after upload")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every export step but skip the HF upload and the dataset "
                         "tag; print what would be uploaded instead")
    ap.add_argument("--check-legacy-tag", action="store_true",
                    help="only check that the dataset is safe to overwrite (scan-legacy "
                         "tagged, or an earlier scan_id export exists); exit 1 if not")
    ap.add_argument("--allow-untagged-legacy", action="store_true",
                    help="upload even though the pre-scan_id state was never tagged "
                         "scan-legacy (its reviews then lose the exact imagery they judged)")
    args = ap.parse_args()

    if args.scan_id is not None and not SCAN_ID_RE.match(args.scan_id):
        ap.error(f"--scan-id {args.scan_id!r} must match {SCAN_ID_RE.pattern!r} "
                  "(AUTOMATION_PLAN.md contract 4.1)")
    args.scan_id = args.scan_id or default_scan_id()

    return args


def _read_results_csv(csv_path: Path) -> pd.DataFrame:
    """Read a results CSV tolerantly, handling mixed 7/8-column formats."""
    try:
        return pd.read_csv(csv_path)
    except pd.errors.ParserError:
        import csv as _csv
        rows = []
        with open(csv_path) as f:
            reader = _csv.reader(f)
            header = next(reader)
            if "input_path" not in header:
                header = header + ["input_path"]
            for row in reader:
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                rows.append(dict(zip(header, row[:len(header)])))
        return pd.DataFrame(rows)


def _is_true(v) -> bool:
    """Robust truthiness — the fallback CSV parser yields the string 'True', not a bool."""
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def _nonempty(v) -> bool:
    return v is not None and str(v).strip() != "" and str(v).strip().lower() != "nan"


def mask_coverage_pct(mask_path: Path) -> float:
    """Share of the tile flagged as windthrow in a proposed mask, in percent.
    Mirrors the frontend's maskCoverage (fraction of nonzero mask pixels) so the
    Expert-Review tab can filter tiles by coverage without decoding each mask."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1) > 0
    return round(float(mask.mean()) * 100.0, 2) if mask.size else 0.0


def encode_input_for_web(src_path: Path, dest_path: Path) -> None:
    """Re-encode a raw float32 model input as a compact uint16 GeoTIFF for the frontend.
    Values outside the representable range are clipped; scale/offset are stored on the bands."""
    with rasterio.open(src_path) as src:
        ms      = src.read()
        profile = src.profile

    dn = np.rint((ms.astype(np.float64) - WEB_OFFSET) / WEB_SCALE)
    dn = np.clip(dn, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    profile.update(
        dtype="uint16", nodata=None, compress="deflate", predictor=2,
        zlevel=9, tiled=False, blockysize=min(profile["height"], 256),
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest_path, "w", **profile) as dst:
        dst.write(dn)
        dst.scales  = [WEB_SCALE]  * profile["count"]
        dst.offsets = [WEB_OFFSET] * profile["count"]


# Per-tile scores the scan records alongside each mask (see src/results.py). They
# are copied into inputs_index.csv so the Expert-Review tab can show and filter on
# model confidence and image quality without downloading anything.
METRIC_COLUMNS = [
    "coverage", "max_patch_px", "clf_prob", "seg_mean_prob_in_mask", "seg_max_prob",
    "valid_frac", "n_scenes", "date_first", "date_last", "cloud_mean",
]


def _metrics(row) -> dict:
    """Copy the metric columns a results.csv carries; missing ones stay empty so
    an index built from an older run keeps the same shape."""
    out = {}
    for col in METRIC_COLUMNS:
        val = row.get(col)
        out[col] = str(val).strip() if _nonempty(val) else ""
    return out


def _job_nbr_encoding(job_dir: Path) -> str:
    """NBR encoding a job wrote its inputs in (its run_meta.json, see
    config.NBR_ENCODING); jobs from before the field existed wrote "rescaled"."""
    try:
        return json.loads((job_dir / "run_meta.json").read_text()).get("nbr_encoding") or "rescaled"
    except (OSError, ValueError):
        return "rescaled"


def _result_rows(csv_path: Path):
    """Yield (tile_id, entry) per row of a results.csv; entry is None for a tile the
    run rejected (negative, or positive but no mask recorded). Rows whose mask file
    is missing are skipped entirely, so a half-written job can't drop a tile."""
    df = _read_results_csv(csv_path)
    nbr_encoding = _job_nbr_encoding(csv_path.parent)
    for _, row in df.iterrows():
        tile_id = row["tile_id"]
        if not _is_true(row.get("is_positive")) or not _nonempty(row.get("mask_path")):
            yield tile_id, None
            continue
        mask_path = ROOT / str(row["mask_path"]).strip()
        if not mask_path.exists():
            continue  # task died mid-write: skip the row rather than export a missing file
        yield tile_id, {
            "mask_path":  mask_path,
            "input_path": ROOT / str(row["input_path"]).strip() if _nonempty(row.get("input_path")) else None,
            "prob_path":  ROOT / str(row["prob_path"]).strip() if _nonempty(row.get("prob_path")) else None,
            "bbox": tuple(float(row[c]) for c in ("lon_min", "lat_min", "lon_max", "lat_max")),
            "metrics": _metrics(row),
            "nbr_encoding": nbr_encoding,
        }


def collect_final_results() -> dict:
    """tile_id -> {"mask_path", "input_path", "bbox"} for every positive tile.

    Starts from the original inference run (outputs/) and lets outputs_resegment/
    override it per tile: a re-segmented tile replaces its original entry, and one
    the new segmenter rejects is dropped. Re-segmentation is optional — with an
    empty outputs_resegment/ the original masks are exported unchanged."""
    final = {}
    for csv_path in sorted(ORIG_DIR.glob("job_*/results.csv")):
        for tile_id, entry in _result_rows(csv_path):
            if entry:
                entry["source"] = "scan"
                final[tile_id] = entry

    for csv_path in sorted(RESEG_DIR.glob("job_*/results.csv")):
        for tile_id, entry in _result_rows(csv_path):
            if entry:
                # Re-segmentation never re-runs the classifier, so keep the stage-1
                # score from the original scan when the new row doesn't carry one.
                prev = final.get(tile_id)
                if prev and not entry["metrics"].get("clf_prob"):
                    entry["metrics"]["clf_prob"] = prev["metrics"].get("clf_prob", "")
                entry["source"] = "resegment"
                final[tile_id] = entry
            else:
                final.pop(tile_id, None)  # re-segmented and rejected

    for entry in final.values():
        entry["kind"] = "flagged"
    return final


def apply_export_rule(flagged: dict) -> tuple:
    """Split the flagged tiles into (detections, dropped) by the export rule
    (src/results.py::is_detection). Each entry's `max_patch_px` comes from
    results.csv or, for a run from before that column, from its mask file."""
    detections, dropped = {}, {}
    for tile_id, entry in flagged.items():
        metrics = entry["metrics"]
        patch = patch_px_of({"max_patch_px": metrics.get("max_patch_px"), "mask_path": str(entry["mask_path"])}, ROOT)
        metrics["max_patch_px"] = str(patch)
        (detections if is_detection(metrics.get("clf_prob"), patch) else dropped)[tile_id] = entry
    return detections, dropped


def check_folder_limit(flagged: dict, n_controls: int) -> None:
    """Exit before anything is staged or uploaded when this round would put
    more than HF_MAX_FILES_PER_DIR files into inputs/ or masks/ (one per
    detection or spot check), with the counts other patch sizes would give."""
    def detections(min_px: int) -> int:
        return sum(1 for e in flagged.values()
                   if passes_classifier(e["metrics"].get("clf_prob")) and int(e["metrics"]["max_patch_px"]) >= min_px)

    n_detections = detections(EXPORT_MIN_PATCH_PX)
    if n_detections + n_controls <= HF_MAX_FILES_PER_DIR:
        return
    print(f"\n  !! {n_detections:,} detections + {n_controls:,} spot checks = {n_detections + n_controls:,} files "
          f"per folder; Hugging Face allows {HF_MAX_FILES_PER_DIR:,}. Detections at other EXPORT_MIN_PATCH_PX:")
    for px in sorted({30, 40, 50, 75, 100, 150, 200, EXPORT_MIN_PATCH_PX}):
        mark = "  <- current" if px == EXPORT_MIN_PATCH_PX else ""
        print(f"       patch >= {px:3d} px ({px * 0.09:4.1f} ha): {detections(px):6,} detections{mark}")
    sys.exit("export stopped, nothing uploaded. Pick a patch size that leaves room for the spot checks, "
             "then: export WINDTHROW_EXPORT_MIN_PATCH_PX=<px> and submit again.")


def collect_control_tiles(final: dict, dropped: Optional[dict] = None) -> dict:
    """tile_id -> entry for every spot-check tile: a tile the scan does not report
    and src/results.py::is_control_sample picks. Those are tiles the classifier
    rejected whose input run_inference kept (is_control), and flagged tiles the
    export rule dropped (`dropped`, from apply_export_rule). The proposed mask is
    empty by definition. A tile that is also in `final` (a detection, e.g. after
    a resumed job) is never a control."""
    controls = {}
    for csv_path in sorted(ORIG_DIR.glob("job_*/results.csv")):
        df = _read_results_csv(csv_path)
        if "is_control" not in df.columns:
            continue   # a run from before spot checks existed
        nbr_encoding = _job_nbr_encoding(csv_path.parent)
        for _, row in df.iterrows():
            tile_id = row["tile_id"]
            if tile_id in final or not _is_true(row.get("is_control")) or not _nonempty(row.get("input_path")):
                continue
            input_path = ROOT / str(row["input_path"]).strip()
            if not input_path.exists():
                continue
            metrics = _metrics(row)
            metrics["coverage"] = metrics["max_patch_px"] = "0"
            controls[tile_id] = {
                "mask_path": None, "input_path": input_path, "prob_path": None,
                "bbox": tuple(float(row[c]) for c in ("lon_min", "lat_min", "lon_max", "lat_max")),
                "metrics": metrics, "source": "scan", "kind": "control",
                "nbr_encoding": nbr_encoding,
            }
    for tile_id, entry in (dropped or {}).items():
        if tile_id in final or not entry["input_path"] or not entry["input_path"].exists():
            continue
        if not is_control_sample(tile_id, entry["metrics"].get("valid_frac")):
            continue
        controls[tile_id] = {
            **entry, "mask_path": None, "prob_path": None, "kind": "control",
            "metrics": dict(entry["metrics"], coverage="0", max_patch_px="0"),
        }
    return controls


def export_geojson(mask_paths: list):
    print(f"Vectorising {len(mask_paths)} masks ...")
    polys = []
    for mp in mask_paths:
        with rasterio.open(mp) as src:
            data      = src.read(1)
            transform = src.transform
        for geom, val in rasterio.features.shapes(data, transform=transform):
            if val == 1:
                polys.append(shape(geom))

    if not polys:
        print("No positive polygons found.")
        return

    gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
    gdf = gdf.dissolve()
    GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(GEOJSON_PATH, driver="GeoJSON")
    print(f"  -> {GEOJSON_PATH}  ({len(gdf)} polygon(s))")


def gdal2tiles_cmd() -> Optional[str]:
    """GDAL's tiler under either of its install names, or None."""
    return shutil.which("gdal2tiles.py") or shutil.which("gdal2tiles")


def export_tiles(mask_paths: list, zoom: str):
    print(f"Building RGBA overlays for {len(mask_paths)} masks ...")
    RGBA_DIR.mkdir(parents=True, exist_ok=True)
    rgba_paths = []
    for mp in mask_paths:
        with rasterio.open(mp) as src:
            mask      = src.read(1)
            profile   = src.profile

        rgba = np.zeros((4, *mask.shape), dtype=np.uint8)
        rgba[0] = 255                    # R
        rgba[3] = (mask * 180).astype(np.uint8)  # alpha: 180 where positive, 0 elsewhere

        out_path = RGBA_DIR / mp.name
        profile.update(count=4, dtype="uint8", nodata=None)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(rgba)
        rgba_paths.append(out_path)

    print(f"Building VRT mosaic ({len(rgba_paths)} tiles) ...")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gdalbuildvrt", "-resolution", "highest", str(VRT_PATH)] + [str(p) for p in rgba_paths],
        check=True,
    )

    print(f"Rendering tile pyramid (z={zoom}) -> {TILES_DIR} ...")
    if TILES_DIR.exists():
        shutil.rmtree(TILES_DIR)
    subprocess.run(
        [gdal2tiles_cmd(), "-p", "mercator", "-z", zoom, "-w", "none", "--xyz",
         "-x", "--processes", str(os.cpu_count() or 4),
         str(VRT_PATH), str(TILES_DIR)],
        check=True,
    )


def _reviewable(results: dict) -> list:
    """(tile_id, entry) for every tile that has a model input to publish — the set
    the Expert-Review tab can actually show."""
    return [(tid, r) for tid, r in results.items() if r["input_path"] and r["input_path"].exists()]


def export_inputs(results: dict):
    """Stage each positive tile's model-input GeoTIFF as inputs/{tile_id}.tif
    (uint16 web encoding, see encode_input_for_web)."""
    entries = _reviewable(results)
    print(f"Staging {len(entries)} input GeoTIFFs ...")
    shutil.rmtree(INPUTS_DIR, ignore_errors=True)   # mirrored upload: only this round's files
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for tile_id, r in entries:
        encode_input_for_web(r["input_path"], INPUTS_DIR / f"{tile_id}.tif")
    print(f"  -> {INPUTS_DIR}  ({len(entries)} file(s))")


def export_index(results: dict, scan_id: str):
    """Write inputs_index.csv: one row per reviewable tile with its bbox and the
    scores the scan recorded for it.

    `coverage` (percent of the tile flagged as windthrow) lets the Expert-Review
    tab filter tiles without decoding each mask; `clf_prob` and the seg_* columns
    carry the model's confidence, `valid_frac`/`cloud_mean`/`date_*` the quality
    and date of the imagery behind the detection. Everything comes from
    results.csv — only `coverage` is recomputed from the mask when an older run
    didn't record it.

    `nbr_encoding` says how the tile's NBR band is stored ("raw" -1..1, or the
    old "rescaled" (NBR + 1) / 2, see config.NBR_ENCODING); absent = rescaled.
    `scan_id` is the last column (AUTOMATION_PLAN.md contract 4.2), the same
    value on every row — readers must tolerate its absence in an older index."""
    entries = _reviewable(results)
    rows = []
    for tile_id, r in entries:
        lon_min, lat_min, lon_max, lat_max = r["bbox"]
        metrics = dict(r.get("metrics") or {})
        if not metrics.get("coverage") and r["mask_path"] and r["mask_path"].exists():
            metrics["coverage"] = mask_coverage_pct(r["mask_path"])
        rows.append({
            "tile_id": tile_id,
            "lon_min": lon_min, "lat_min": lat_min,
            "lon_max": lon_max, "lat_max": lat_max,
            **{col: metrics.get(col, "") for col in METRIC_COLUMNS},
            "source": r.get("source", ""),
            "kind": r.get("kind", "flagged"),
            "nbr_encoding": r.get("nbr_encoding", "rescaled"),
            "scan_id": scan_id,
        })

    if not rows:
        # An empty index would take every tile out of the Expert-Review tab. That
        # is almost always a missing/cleared outputs/ rather than a real result,
        # so leave whatever is published in place and say so.
        print("  !! no reviewable tiles found — keeping the existing inputs_index.csv")
        return

    INPUTS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(INPUTS_INDEX_PATH, index=False)
    print(f"  -> {INPUTS_INDEX_PATH}  ({len(rows)} row(s))")


def export_probs(results: dict):
    """Stage each positive tile's segmenter probability map as probs/{tile_id}.tif
    (uint8, probability = DN/255). Optional: the review UI does not need it, but
    publishing it lets a client re-threshold a detection without the GPU pass."""
    entries = [(tid, r) for tid, r in results.items() if r.get("prob_path") and r["prob_path"].exists()]
    print(f"Staging {len(entries)} probability maps ...")
    shutil.rmtree(PROBS_DIR, ignore_errors=True)   # mirrored upload: only this round's files
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    for tile_id, r in entries:
        shutil.copyfile(r["prob_path"], PROBS_DIR / f"{tile_id}.tif")
    print(f"  -> {PROBS_DIR}  ({len(entries)} file(s))")


def export_masks(results: dict):
    """Stage each positive tile's proposed mask as masks/{tile_id}.tif — a single-band
    uint8 (0/1) GeoTIFF on the same grid as inputs/{tile_id}.tif, so the Expert-Review
    tab can load a per-pixel-editable proposed mask that lines up with the input imagery."""
    entries = [
        (tid, r) for tid, r in results.items()
        if (r["mask_path"] and r["mask_path"].exists())
        or (r.get("kind") == "control" and r["input_path"] and r["input_path"].exists())
    ]
    print(f"Staging {len(entries)} mask GeoTIFFs ...")
    shutil.rmtree(MASKS_DIR, ignore_errors=True)   # mirrored upload: only this round's files
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    for tile_id, r in entries:
        if r["mask_path"]:
            with rasterio.open(r["mask_path"]) as src:
                mask    = (src.read(1) > 0).astype("uint8")
                profile = src.profile
        else:
            # Spot check: the model proposes nothing, so an all-zero mask on the
            # input's own grid — the review tab loads it like any other.
            with rasterio.open(r["input_path"]) as src:
                mask    = np.zeros((src.height, src.width), dtype="uint8")
                profile = src.profile
        profile.update(count=1, dtype="uint8", nodata=None, compress="deflate")
        with rasterio.open(MASKS_DIR / f"{tile_id}.tif", "w", **profile) as dst:
            dst.write(mask, 1)
    print(f"  -> {MASKS_DIR}  ({len(entries)} file(s))")


def _model_block() -> dict:
    """{"classifier": {...}, "segmenter": {...}} for scan_meta.json (contract
    4.3), copied from MODEL_VERSIONS_FILE (written once per scan by
    scripts/resolve_model_versions.py). Either value is None — with a printed
    warning — if that file is missing, e.g. a manual export run without step 00.

    If outputs_resegment/ has any results (step 99 ran), the segmenter block is
    overridden with the *local* checkpoint recorded in its run_meta.json: that
    step never touches HF, so `revision` becomes "local:<path>" rather than a
    commit sha, and `repo_id` is None."""
    if not MODEL_VERSIONS_FILE.exists():
        print(f"  !! {MODEL_VERSIONS_FILE.name} not found — scan_meta.json model block will be null")
        return {"classifier": None, "segmenter": None}

    versions = json.loads(MODEL_VERSIONS_FILE.read_text())
    classifier = versions.get("classifier")
    segmenter  = versions.get("segmenter")
    classifier_block = (
        {"repo_id": classifier["repo_id"], "revision": classifier["revision"]}
        if classifier else None
    )
    segmenter_block = (
        {"repo_id": segmenter["repo_id"], "revision": segmenter["revision"]}
        if segmenter else None
    )

    reseg_meta_paths = sorted(RESEG_DIR.glob("job_*/run_meta.json"))
    if reseg_meta_paths:
        reseg_meta = json.loads(reseg_meta_paths[0].read_text())
        segmenter_block = {
            "repo_id": None,
            "revision": f"local:{reseg_meta['segmenter_path']}",
        }

    return {"classifier": classifier_block, "segmenter": segmenter_block}


def _data_sources() -> list:
    """Imagery source(s) the inference run used ("gee" / "pc"), from the
    per-job run_meta.json files; runs from before the field existed were GEE."""
    found = set()
    for path in ORIG_DIR.glob("job_*/run_meta.json"):
        try:
            found.add(json.loads(path.read_text()).get("data_source") or "gee")
        except (OSError, ValueError):
            continue
    return sorted(found)


def write_scan_meta(scan_id: str, n_positive_tiles: int, n_control_tiles: int = 0,
                    nbr_encodings: Optional[list] = None, n_flagged_tiles: Optional[int] = None) -> None:
    """Write scan_meta.json at the export root (contract 4.3): which pinned
    models produced this scan_id, from which imagery source, and how many
    positive tiles it carries, so a downstream reader (training's manifest, a
    human comparing rounds) doesn't have to reconstruct it from
    model_versions.json + run_meta.json by hand."""
    meta = {
        "scan_id": scan_id,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **_model_block(),
        "data_source": _data_sources(),
        "nbr_encoding": nbr_encodings or [],
        "export_rule": {"min_clf_prob": EXPORT_MIN_CLF_PROB, "min_patch_px": EXPORT_MIN_PATCH_PX},
        "n_flagged_tiles": n_flagged_tiles,    # flagged by the classifier
        "n_positive_tiles": n_positive_tiles,  # of those, detections exported
        "n_control_tiles": n_control_tiles,
    }
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"  -> {SCAN_META_PATH}")


# Upload order (contract, AUTOMATION_PLAN.md T2 step 4): geojson -> tiles ->
# inputs -> masks -> probs -> scan_meta.json -> inputs_index.csv *last*.
# inputs_index.csv is what makes a tile visible to the Expert-Review tab, so it
# must go up only after that tile's imagery and mask are already there —
# otherwise a reviewer can briefly see a new tile with no image, or an old
# mask under a refreshed index (F7 in AUTOMATION_PLAN.md).
# Remote folders mirrored exactly, not just added to: remote files the local
# folder no longer has are deleted (scoped to that folder) in a commit right
# before the upload, see delete_stale_remote. Reviewers can hit missing
# images of the previous round until inputs_index.csv lands, so pause
# reviewing during an export.
# tiles/ needs it because gdal2tiles -x writes no tile where a scan found
# nothing, so without it a detection the new model dropped would stay on the
# map forever. inputs/, masks/ and probs/ need it because Hugging Face allows
# at most HF_MAX_FILES_PER_DIR files per folder: keeping every round's tiles
# on main would hit that. Earlier rounds' files stay reachable at their
# scan_id tags, which is where windthrowAI-training reads them from.
MIRRORED_FOLDERS = {"tiles", "inputs", "masks", "probs"}


def _upload_plan(args):
    """[(local_path, path_in_repo, enabled, is_folder), ...] in upload order.
    Shared by upload_to_hf (real) and print_upload_plan (--dry-run) so the two
    can never drift apart."""
    return [
        (GEOJSON_PATH,       "windthrow.geojson", True,                False),
        (TILES_DIR,          "tiles",             not args.skip_tiles, True),
        (INPUTS_DIR,         "inputs",            not args.skip_inputs, True),
        (MASKS_DIR,          "masks",             not args.skip_masks, True),
        (PROBS_DIR,          "probs",             args.include_probs, True),
        (SCAN_META_PATH,     "scan_meta.json",    True,                False),
        (INPUTS_INDEX_PATH,  "inputs_index.csv",  not args.skip_index, False),
    ]


def delete_stale_remote(api, repo_id: str, folder: Path, path_in_repo: str) -> int:
    """Delete the remote files under path_in_repo/ that the local `folder` does
    not have, in one commit, *before* that folder is uploaded. upload_folder's
    own delete_patterns can't be relied on for this: recent huggingface_hub
    versions upload a large folder in several commits, and unless the deletions
    land first, an intermediate commit holds old + new files and Hugging Face
    rejects it past HF_MAX_FILES_PER_DIR. Returns how many files were removed."""
    from huggingface_hub import CommitOperationDelete

    prefix = path_in_repo.strip("/") + "/"
    local = {prefix + p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    stale = [f for f in api.list_repo_files(repo_id, repo_type="dataset")
             if f.startswith(prefix) and f not in local]
    if stale:
        api.create_commit(repo_id=repo_id, repo_type="dataset",
                          operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
                          commit_message=f"Remove {len(stale)} files of earlier rounds from {prefix}")
    return len(stale)


def upload_to_hf(repo_id: str, args):
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Uploading to HF dataset repo: {repo_id}")

    for path, path_in_repo, enabled, is_folder in _upload_plan(args):
        if not enabled or not path.exists():
            continue
        if is_folder:
            if path_in_repo in MIRRORED_FOLDERS:
                n = delete_stale_remote(api, repo_id, path, path_in_repo)
                print(f"  removed {n} files of earlier rounds from {path_in_repo}/")
            api.upload_folder(folder_path=str(path), path_in_repo=path_in_repo,
                               repo_id=repo_id, repo_type="dataset")
            print(f"  uploaded {path_in_repo}/")
        else:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo,
                             repo_id=repo_id, repo_type="dataset")
            print(f"  uploaded {path_in_repo}")


def legacy_state_safe(repo_id: str) -> bool:
    """Whether overwriting inputs/ and masks/ keeps every review traceable:
    `scan-legacy` is tagged, or an earlier scan_id export already tagged its
    round (the legacy state is then whatever it is — nothing left to save).
    Prints what to do when it isn't, or when the tags can't be read."""
    from huggingface_hub import HfApi
    try:
        tags = {t.name for t in HfApi().list_repo_refs(repo_id, repo_type="dataset").tags}
    except Exception as exc:  # noqa: BLE001 — unknown is not safe
        print(f"  !! could not read the tags of {repo_id}: {exc}")
        return False
    if LEGACY_SCAN_ID in tags:
        return True
    if any(SCAN_ID_RE.match(t) for t in tags):
        print(f"  ! {repo_id} has no {LEGACY_SCAN_ID} tag, but earlier scan_id exports are tagged; "
              "the pre-scan_id state is already gone, so there is nothing left to protect.")
        return True
    print(
        f"  !! {repo_id} has no {LEGACY_SCAN_ID} tag. This export overwrites inputs/ and masks/, and\n"
        "     the reviews made so far point at that tag for the imagery they judged. Tag it once:\n"
        "       python -c \"from huggingface_hub import HfApi; HfApi().create_tag("
        f"'{repo_id}', tag='{LEGACY_SCAN_ID}', repo_type='dataset', "
        "tag_message='State reviewed before scan_id existed')\""
    )
    return False


def tag_dataset(repo_id: str, scan_id: str):
    """Tag the dataset repo at the commit the export just uploaded, so
    `revision=scan_id` reproduces exactly what reviewers saw during this round
    (contract 4.1). Called only after every upload above has succeeded."""
    from huggingface_hub import HfApi

    HfApi().create_tag(repo_id, tag=scan_id, repo_type="dataset")
    print(f"  tagged {repo_id} @ {scan_id}")


def print_upload_plan(args, scan_id: str):
    """--dry-run stand-in for upload_to_hf + tag_dataset: touches no HF state,
    just prints the same order so the plan is visible without a write token."""
    print(f"[dry-run] would upload to HF dataset repo: {args.hf_repo}")
    for path, path_in_repo, enabled, is_folder in _upload_plan(args):
        if not enabled:
            continue
        if not path.exists():
            print(f"  [dry-run] would skip {path_in_repo} (not found)")
            continue
        if is_folder:
            n = sum(1 for p in path.rglob("*") if p.is_file())
            mirror = ", deleting remote files not in it" if path_in_repo in MIRRORED_FOLDERS else ""
            print(f"  [dry-run] would upload {path_in_repo}/  ({n} file(s){mirror})")
        else:
            print(f"  [dry-run] would upload {path_in_repo}")

    if args.no_tag:
        print(f"  [dry-run] --no-tag: would skip tagging {scan_id}")
    else:
        print(f"  [dry-run] would tag dataset @ {scan_id}")


def main():
    args = parse_args()
    if args.check_legacy_tag:
        sys.exit(0 if legacy_state_safe(args.hf_repo) else 1)
    if not args.dry_run and not args.allow_untagged_legacy and not legacy_state_safe(args.hf_repo):
        sys.exit("refusing to export (see above), nothing was uploaded; "
                 "--allow-untagged-legacy overrides")
    if not args.skip_tiles and not (shutil.which("gdalbuildvrt") and gdal2tiles_cmd()):
        sys.exit("the map overlay needs GDAL's command-line tools (gdalbuildvrt, gdal2tiles.py), "
                 "which are not on PATH. Once: `conda create -n gdal -c conda-forge gdal` "
                 "(slurm/03 picks that env up). Or publish without refreshing the map: --skip-tiles")
    scan_id = args.scan_id
    print(f"scan_id: {scan_id}")

    flagged = collect_final_results()
    final_results, dropped = apply_export_rule(flagged)
    print(f"Flagged by the classifier (resegmented overrides original): {len(flagged)}")
    print(f"Detections exported (clf_prob >= {EXPORT_MIN_CLF_PROB}, patch >= {EXPORT_MIN_PATCH_PX} px): "
          f"{len(final_results)}; {len(dropped)} flagged tiles below the rule")
    mask_paths = [r["mask_path"] for r in final_results.values()]
    controls = collect_control_tiles(final_results, dropped)
    print(f"Spot-check tiles (not reported, sampled for review): {len(controls)}")
    reviewable = {**final_results, **controls}   # the map and GeoJSON stay detections-only
    check_folder_limit(flagged, len(controls))

    export_geojson(mask_paths)
    if not args.skip_tiles:
        export_tiles(mask_paths, args.zoom)
    if not args.skip_inputs:
        export_inputs(reviewable)
    if not args.skip_index:
        export_index(reviewable, scan_id)
    if not args.skip_masks:
        export_masks(reviewable)
    if args.include_probs:
        export_probs(final_results)

    write_scan_meta(scan_id, len(final_results), len(controls),
                    sorted({r.get("nbr_encoding", "rescaled") for r in reviewable.values()}), len(flagged))

    if args.dry_run:
        print_upload_plan(args, scan_id)
    else:
        upload_to_hf(args.hf_repo, args)
        if not args.no_tag:
            tag_dataset(args.hf_repo, scan_id)
        else:
            print("  --no-tag: skipped dataset tag")

    print("Done.")


if __name__ == "__main__":
    main()

"""
Build the final windthrow result set and push it to the Hugging Face dataset
used by the frontend, as both a vector layer (GeoJSON) and a raster tile
pyramid (PNG, Web Mercator z/x/y) for map overlay.

    python scripts/export_for_frontend.py

Steps:
  1. Pick the final mask per tile: outputs/ (original run), overridden per tile by
     outputs_resegment/ (new segmenter) where it exists. Re-segmentation is optional.
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

Steps 4-6 also cover the spot-check tiles: rejected tiles run_inference kept an
input for (is_control). They get an all-zero mask and `kind=control` in the
index, and stay out of the GeoJSON and the map (steps 2-3), which show
detections only. Every other index row is `kind=flagged`.
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

Requires on PATH: gdalbuildvrt, gdal2tiles.py (part of GDAL).
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

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODEL_VERSIONS_FILE

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
    "coverage", "clf_prob", "seg_mean_prob_in_mask", "seg_max_prob",
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


def _result_rows(csv_path: Path):
    """Yield (tile_id, entry) per row of a results.csv; entry is None for a tile the
    run rejected (negative, or positive but no mask recorded). Rows whose mask file
    is missing are skipped entirely, so a half-written job can't drop a tile."""
    df = _read_results_csv(csv_path)
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


def collect_control_tiles(final: dict) -> dict:
    """tile_id -> entry for every spot-check tile of the original run: a tile the
    classifier rejected whose input run_inference kept (is_control, see
    src/results.py::is_control_sample). No mask on disk — the proposed mask is
    empty by definition. A tile that is also in `final` (positive after all,
    e.g. a resumed job) is a detection, not a control."""
    controls = {}
    for csv_path in sorted(ORIG_DIR.glob("job_*/results.csv")):
        df = _read_results_csv(csv_path)
        if "is_control" not in df.columns:
            continue   # a run from before spot checks existed
        for _, row in df.iterrows():
            tile_id = row["tile_id"]
            if tile_id in final or not _is_true(row.get("is_control")) or not _nonempty(row.get("input_path")):
                continue
            input_path = ROOT / str(row["input_path"]).strip()
            if not input_path.exists():
                continue
            metrics = _metrics(row)
            metrics["coverage"] = "0"
            controls[tile_id] = {
                "mask_path": None, "input_path": input_path, "prob_path": None,
                "bbox": tuple(float(row[c]) for c in ("lon_min", "lat_min", "lon_max", "lat_max")),
                "metrics": metrics, "source": "scan", "kind": "control",
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
        ["gdal2tiles.py", "-p", "mercator", "-z", zoom, "-w", "none", "--xyz",
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


def write_scan_meta(scan_id: str, n_positive_tiles: int, n_control_tiles: int = 0) -> None:
    """Write scan_meta.json at the export root (contract 4.3): which pinned
    models produced this scan_id, and how many positive tiles it carries, so a
    downstream reader (training's manifest, a human comparing rounds) doesn't
    have to reconstruct it from model_versions.json + run_meta.json by hand."""
    meta = {
        "scan_id": scan_id,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **_model_block(),
        "n_positive_tiles": n_positive_tiles,
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
# folder no longer has are deleted in the same commit (scoped to that folder).
# tiles/ needs it because gdal2tiles -x writes no tile where a scan found
# nothing, so without it a detection the new model dropped would stay on the
# map forever. inputs/ and masks/ don't: a tile missing from inputs_index.csv
# is already invisible to reviewers.
MIRRORED_FOLDERS = {"tiles"}


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


def upload_to_hf(repo_id: str, args):
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Uploading to HF dataset repo: {repo_id}")

    for path, path_in_repo, enabled, is_folder in _upload_plan(args):
        if not enabled or not path.exists():
            continue
        if is_folder:
            mirrored = path_in_repo in MIRRORED_FOLDERS
            api.upload_folder(folder_path=str(path), path_in_repo=path_in_repo,
                               repo_id=repo_id, repo_type="dataset",
                               delete_patterns="*" if mirrored else None)
            print(f"  uploaded {path_in_repo}/" + (" (stale remote files removed)" if mirrored else ""))
        else:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo,
                             repo_id=repo_id, repo_type="dataset")
            print(f"  uploaded {path_in_repo}")


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
    scan_id = args.scan_id
    print(f"scan_id: {scan_id}")

    final_results = collect_final_results()
    print(f"Final mask count (resegmented overrides original): {len(final_results)}")
    mask_paths = [r["mask_path"] for r in final_results.values()]
    controls = collect_control_tiles(final_results)
    print(f"Spot-check tiles (rejected, sampled for review): {len(controls)}")
    reviewable = {**final_results, **controls}   # the map and GeoJSON stay detections-only

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

    write_scan_meta(scan_id, len(final_results), len(controls))

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

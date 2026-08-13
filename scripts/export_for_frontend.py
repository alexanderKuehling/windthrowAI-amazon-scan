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
     (encode_input_for_web) + a tile_id->bbox
     index CSV (for lookup by location) that also carries each tile's windthrow
     `coverage` (%) so the Expert-Review tab can filter by coverage.
  5. Stage each positive tile's proposed mask as a single-band uint8 GeoTIFF
     (masks/{tile_id}.tif), aligned to inputs/{tile_id}.tif, for the Expert-Review tab.
  6. Upload geojson + tiles/ + inputs/ + inputs_index.csv + masks/ to the HF dataset repo.

Requires on PATH: gdalbuildvrt, gdal2tiles.py (part of GDAL).
Requires Python packages: rasterio, geopandas, shapely, huggingface_hub.
Requires `huggingface-cli login` (or HF_TOKEN env var) with write access to
the target dataset repo.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESEG_DIR    = ROOT / "outputs_resegment"
ORIG_DIR     = ROOT / "outputs"
WORK_DIR     = ROOT / "outputs_final"
RGBA_DIR     = WORK_DIR / "rgba_masks"
TILES_DIR    = WORK_DIR / "tiles"
INPUTS_DIR   = WORK_DIR / "inputs"
MASKS_DIR    = WORK_DIR / "masks"
VRT_PATH     = WORK_DIR / "mosaic.vrt"
GEOJSON_PATH = WORK_DIR / "windthrow.geojson"
INPUTS_INDEX_PATH = WORK_DIR / "inputs_index.csv"

HF_REPO_ID   = "alexKlg/windthrowAI-dataset-segmentation_results"

# Web encoding of the model inputs: reflectance = DN * WEB_SCALE + WEB_OFFSET.
# uint16 + deflate is ~3.5x smaller than the raw float32/LZW tif (see encode_input_for_web).
WEB_SCALE  = 1e-4
WEB_OFFSET = -1.0   # DN 0..65535 -> -1.0 .. 5.5535 (covers negative NBR)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=HF_REPO_ID)
    ap.add_argument("--zoom",    default="5-13", help="gdal2tiles.py -z range")
    ap.add_argument("--skip-tiles", action="store_true", help="only export GeoJSON, skip raster tiles")
    ap.add_argument("--skip-inputs", action="store_true", help="skip exporting model-input GeoTIFFs")
    ap.add_argument("--skip-masks", action="store_true", help="skip exporting per-tile mask GeoTIFFs")
    return ap.parse_args()


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
            "bbox": tuple(float(row[c]) for c in ("lon_min", "lat_min", "lon_max", "lat_max")),
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
                final[tile_id] = entry

    for csv_path in sorted(RESEG_DIR.glob("job_*/results.csv")):
        for tile_id, entry in _result_rows(csv_path):
            if entry:
                final[tile_id] = entry
            else:
                final.pop(tile_id, None)  # re-segmented and rejected

    return final


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


def export_inputs(results: dict):
    """Stage each positive tile's model-input GeoTIFF as inputs/{tile_id}.tif (uint16
    web encoding, see encode_input_for_web), plus an index CSV of tile_id -> bbox +
    windthrow coverage. The `coverage` column (percent of the tile flagged as windthrow
    in the proposed mask) lets the Expert-Review tab filter tiles by coverage without
    decoding each mask."""
    entries = [(tid, r) for tid, r in results.items() if r["input_path"] and r["input_path"].exists()]
    print(f"Staging {len(entries)} input GeoTIFFs ...")
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tile_id, r in entries:
        dest = INPUTS_DIR / f"{tile_id}.tif"
        encode_input_for_web(r["input_path"], dest)
        lon_min, lat_min, lon_max, lat_max = r["bbox"]
        coverage = mask_coverage_pct(r["mask_path"]) if r["mask_path"] and r["mask_path"].exists() else ""
        rows.append({"tile_id": tile_id, "lon_min": lon_min, "lat_min": lat_min,
                     "lon_max": lon_max, "lat_max": lat_max, "coverage": coverage})

    pd.DataFrame(rows).to_csv(INPUTS_INDEX_PATH, index=False)
    print(f"  -> {INPUTS_DIR}  ({len(entries)} file(s))")
    print(f"  -> {INPUTS_INDEX_PATH}")


def export_masks(results: dict):
    """Stage each positive tile's proposed mask as masks/{tile_id}.tif — a single-band
    uint8 (0/1) GeoTIFF on the same grid as inputs/{tile_id}.tif, so the Expert-Review
    tab can load a per-pixel-editable proposed mask that lines up with the input imagery."""
    entries = [(tid, r) for tid, r in results.items() if r["mask_path"] and r["mask_path"].exists()]
    print(f"Staging {len(entries)} mask GeoTIFFs ...")
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    for tile_id, r in entries:
        with rasterio.open(r["mask_path"]) as src:
            mask    = (src.read(1) > 0).astype("uint8")
            profile = src.profile
        profile.update(count=1, dtype="uint8", nodata=None, compress="deflate")
        with rasterio.open(MASKS_DIR / f"{tile_id}.tif", "w", **profile) as dst:
            dst.write(mask, 1)
    print(f"  -> {MASKS_DIR}  ({len(entries)} file(s))")


def upload_to_hf(repo_id: str, skip_tiles: bool, skip_inputs: bool, skip_masks: bool):
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Uploading to HF dataset repo: {repo_id}")

    if GEOJSON_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(GEOJSON_PATH),
            path_in_repo="windthrow.geojson",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  uploaded windthrow.geojson")

    if not skip_tiles and TILES_DIR.exists():
        api.upload_folder(
            folder_path=str(TILES_DIR),
            path_in_repo="tiles",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  uploaded tiles/")

    if not skip_inputs and INPUTS_DIR.exists():
        api.upload_folder(
            folder_path=str(INPUTS_DIR),
            path_in_repo="inputs",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  uploaded inputs/")

    if not skip_inputs and INPUTS_INDEX_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(INPUTS_INDEX_PATH),
            path_in_repo="inputs_index.csv",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  uploaded inputs_index.csv")

    if not skip_masks and MASKS_DIR.exists():
        api.upload_folder(
            folder_path=str(MASKS_DIR),
            path_in_repo="masks",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  uploaded masks/")


def main():
    args = parse_args()

    final_results = collect_final_results()
    print(f"Final mask count (resegmented overrides original): {len(final_results)}")
    mask_paths = [r["mask_path"] for r in final_results.values()]

    export_geojson(mask_paths)
    if not args.skip_tiles:
        export_tiles(mask_paths, args.zoom)
    if not args.skip_inputs:
        export_inputs(final_results)
    if not args.skip_masks:
        export_masks(final_results)

    upload_to_hf(args.hf_repo, args.skip_tiles, args.skip_inputs, args.skip_masks)
    print("Done.")


if __name__ == "__main__":
    main()

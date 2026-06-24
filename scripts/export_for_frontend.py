"""
Build the final windthrow result set and push it to the Hugging Face dataset
used by the frontend, as both a vector layer (GeoJSON) and a raster tile
pyramid (PNG, Web Mercator z/x/y) for map overlay.

    python scripts/export_for_frontend.py

Steps:
  1. Pick the final mask per tile: prefer outputs_resegment/ (new segmenter)
     over outputs/ (original run) when both exist.
  2. Vectorize all final masks into one dissolved GeoJSON.
  3. Convert each mask to a red/transparent RGBA GeoTIFF, build a VRT mosaic,
     and render a PNG tile pyramid with gdal2tiles.py.
  4. Stage each positive tile's raw 7-band model input GeoTIFF + a tile_id->bbox
     index CSV (for later lookup by location).
  5. Upload geojson + tiles/ + inputs/ + inputs_index.csv to the HF dataset repo.

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
VRT_PATH     = WORK_DIR / "mosaic.vrt"
GEOJSON_PATH = WORK_DIR / "windthrow.geojson"
INPUTS_INDEX_PATH = WORK_DIR / "inputs_index.csv"

HF_REPO_ID   = "alexKlg/windthrowAI-dataset-segmentation_results"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=HF_REPO_ID)
    ap.add_argument("--zoom",    default="5-13", help="gdal2tiles.py -z range")
    ap.add_argument("--skip-tiles", action="store_true", help="only export GeoJSON, skip raster tiles")
    ap.add_argument("--skip-inputs", action="store_true", help="skip exporting model-input GeoTIFFs")
    return ap.parse_args()


def collect_final_results() -> dict:
    """tile_id -> {"mask_path", "input_path", "bbox"}, resegmented results only."""
    final = {}
    for csv_path in sorted(RESEG_DIR.glob("job_*/results.csv")):
        df = pd.read_csv(csv_path)
        for _, row in df[df["is_positive"] == True].iterrows():
            if row["mask_path"]:
                final[row["tile_id"]] = {
                    "mask_path":  ROOT / row["mask_path"],
                    "input_path": ROOT / row["input_path"] if row.get("input_path") else None,
                    "bbox": (row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"]),
                }

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
    """Stage each positive tile's raw model-input GeoTIFF as inputs/{tile_id}.tif,
    plus an index CSV of tile_id -> bbox so a tile can be looked up by location."""
    entries = [(tid, r) for tid, r in results.items() if r["input_path"] and r["input_path"].exists()]
    print(f"Staging {len(entries)} input GeoTIFFs ...")
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tile_id, r in entries:
        dest = INPUTS_DIR / f"{tile_id}.tif"
        shutil.copyfile(r["input_path"], dest)
        lon_min, lat_min, lon_max, lat_max = r["bbox"]
        rows.append({"tile_id": tile_id, "lon_min": lon_min, "lat_min": lat_min,
                     "lon_max": lon_max, "lat_max": lat_max})

    pd.DataFrame(rows).to_csv(INPUTS_INDEX_PATH, index=False)
    print(f"  -> {INPUTS_DIR}  ({len(entries)} file(s))")
    print(f"  -> {INPUTS_INDEX_PATH}")


def upload_to_hf(repo_id: str, skip_tiles: bool, skip_inputs: bool):
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

    upload_to_hf(args.hf_repo, args.skip_tiles, args.skip_inputs)
    print("Done.")


if __name__ == "__main__":
    main()

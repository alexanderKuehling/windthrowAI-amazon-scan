"""
Merge all per-job result CSVs and build a VRT mosaic of positive-tile masks.

Run after all inference jobs complete:
    python scripts/merge_results.py

Outputs:
    outputs/all_results.csv          – full tile-level results
    outputs/windthrow_amazon.vrt     – virtual mosaic (open in QGIS / gdal)
    outputs/windthrow_amazon.geojson – vectorised windthrow polygons

"""

import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.features
from shapely.geometry import shape

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import GRID_FILE, OUTPUTS_DIR, TILES_PER_JOB
from src.tile_grid import n_jobs



def main():
    # ── Collect all job CSVs ─────────────────────────────────────────────
    csv_files = sorted(OUTPUTS_DIR.glob("job_*/results.csv"))
    if not csv_files:
        print("No result CSVs found. Run inference first.")
        return

    expected = n_jobs(GRID_FILE, TILES_PER_JOB)
    if len(csv_files) != expected:
        sys.exit(
            f"expected {expected} job dirs (per tile_grid.parquet), found "
            f"{len(csv_files)} — refusing to merge a partial run. Check "
            "slurm_logs/infer/ for failed array tasks and resubmit them "
            "(sbatch --array=<idx> slurm/01_inference_array.sh) before merging."
        )

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df.to_csv(OUTPUTS_DIR / "all_results.csv", index=False)
    print(f"Total tiles processed : {len(df):,}")
    print(f"Positive (windthrow)  : {df['is_positive'].sum():,}")
    print(f"Download failures     : {(~df['download_ok']).sum():,}")

    # ── Build VRT of all mask GeoTIFFs ───────────────────────────────────
    mask_paths = df.loc[df["is_positive"] & (df["mask_path"] != ""), "mask_path"].tolist()
    abs_masks  = [str(ROOT / p) for p in mask_paths]

    if abs_masks:
        vrt_path = str(OUTPUTS_DIR / "windthrow_amazon.vrt")
        subprocess.run(
            ["gdalbuildvrt", "-resolution", "highest", vrt_path] + abs_masks,
            check=True
        )
        print(f"VRT mosaic written → {vrt_path}")

        # ── Vectorise masks to GeoJSON ───────────────────────────────────
        polys = []
        for mp in abs_masks:
            with rasterio.open(mp) as src:
                data = src.read(1)
                transform = src.transform
                crs = src.crs
            for geom, val in rasterio.features.shapes(data, transform=transform):
                if val == 1:
                    polys.append(shape(geom))

        if polys:
            gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
            gdf = gdf.dissolve()          # merge touching polygons
            out_geojson = OUTPUTS_DIR / "windthrow_amazon.geojson"
            gdf.to_file(out_geojson, driver="GeoJSON")
            print(f"Vector output → {out_geojson}  ({len(gdf)} polygon(s))")
    else:
        print("No positive tiles found – nothing to mosaic.")


if __name__ == "__main__":
    main()

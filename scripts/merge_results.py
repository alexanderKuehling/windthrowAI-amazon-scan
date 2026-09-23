"""
Merge all per-job result CSVs and build a VRT mosaic of positive-tile masks.

Run after all inference jobs complete:
    python scripts/merge_results.py

Outputs:
    outputs/all_results.csv          – full tile-level results
    outputs/windthrow_amazon.vrt     – virtual mosaic (open in QGIS / gdal)
    outputs/windthrow_amazon.geojson – vectorised windthrow polygons

At the end, best-effort compares this run's positive count against the
inputs_index.csv currently published on HF and warns loudly if it moved by
more than 50% — the only guard against a classifier fine-tuned into flagging
almost everything (AUTOMATION_PLAN.md F7/T8). Advisory only: a download
failure here (offline, no prior export, no token) only prints a note and
never fails the merge.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.features
from shapely.geometry import shape

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import GRID_FILE, OUTPUTS_DIR, TILES_PER_JOB
from src.tile_grid import n_jobs

# Same dataset scripts/export_for_frontend.py publishes to. Declared here
# rather than imported so this script doesn't pull in that module's GDAL-tiling
# machinery just to read one constant (mirrors summarize_reviews.py's HF_REPO_ID).
HF_DATASET_REPO = "alexKlg/windthrowAI-dataset-segmentation_results"


def _published_positive_count(repo_id: str) -> Optional[int]:
    """Row count of the currently-published inputs_index.csv (one row per
    positive tile), for a sanity check against this run's positive count.

    Returns None on any failure — no network, no prior export, no token — so a
    download problem here can only skip the check below; it must never fail
    the merge (AUTOMATION_PLAN.md T8)."""
    try:
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
        path = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                filename="inputs_index.csv", token=token)
        with open(path, newline="") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception as exc:  # noqa: BLE001 — this check is best-effort only
        print(f"  (could not fetch previous inputs_index.csv for comparison: {exc})")
        return None


def _print_positive_delta(n_new: int) -> None:
    """Compare this run's positive count against the last published index —
    the only guard (until it is reviewed) against a classifier that got
    fine-tuned into flagging almost everything. Advisory only: it prints a
    line and, if the swing is large, a loud warning; it never blocks."""
    n_old = _published_positive_count(HF_DATASET_REPO)
    if n_old is None:
        return

    if n_old == 0:
        delta_pct = 0.0 if n_new == 0 else float("inf")
    else:
        delta_pct = (n_new - n_old) / n_old * 100

    print(f"positives: {n_old:,} -> {n_new:,} ({delta_pct:+.1f}%)")
    if abs(delta_pct) > 50:
        print(
            f"  !! positive count changed by {delta_pct:+.1f}% since the last "
            "published inputs_index.csv — check the classifier before "
            "exporting/publishing this run"
        )


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

    _print_positive_delta(int(df["is_positive"].sum()))


if __name__ == "__main__":
    main()

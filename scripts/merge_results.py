"""
Merge all per-job result CSVs and build a VRT mosaic of the detections' masks.

Run after all inference jobs complete:
    python scripts/merge_results.py

Outputs:
    outputs/all_results.csv          – full tile-level results
    outputs/windthrow_amazon.vrt     – virtual mosaic of the detections (open in QGIS / gdal)
    outputs/windthrow_amazon.geojson – their vectorised windthrow polygons

Before publishing, the round has to pass three checks. Any failure exits
non-zero *after* writing the outputs above, and that cancels the chained export
(03, --kill-on-invalid-dep), so reviewers never see a broken round:

  missing tiles    a grid tile has no row at all (an array task died mid-run)
  download errors  more than MERGE_MAX_ERROR_FRACTION of the tiles with imagery
                   failed (quota, timeouts) and would be missing from the round
  positive count   the detections (flagged tiles passing the export rule,
                   src/results.py::is_detection) moved by more than
                   MERGE_MAX_POSITIVE_DELTA against the inputs_index.csv on HF —
                   a classifier fine-tuned into flagging almost everything, or a
                   broken download (AUTOMATION_PLAN.md T8)

slurm/resume_scan.sh resubmits exactly the unfinished array tasks and chains
02 -> 03 again; --allow-errors / --accept-delta (forwarded by it) accept a
round on purpose. The positive-count check is skipped, with a note, when the
published index can't be fetched. `--list-incomplete` only prints the array
task indexes that still have missing or failed tiles (used by resume_scan.sh).
"""

import argparse
import csv
import os
import shutil
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

from src.config import (
    GRID_FILE,
    MERGE_MAX_ERROR_FRACTION,
    MERGE_MAX_POSITIVE_DELTA,
    OUTPUTS_DIR,
    TILES_PER_JOB,
)
from src.results import is_detection, is_retryable, patch_px_of
from src.tile_grid import load_grid, n_jobs

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
            # spot-check rows (kind=control) are rejected tiles, not detections
            return sum(1 for r in csv.DictReader(f) if r.get("kind") != "control")
    except Exception as exc:  # noqa: BLE001 — this check is best-effort only
        print(f"  (could not fetch previous inputs_index.csv for comparison: {exc})")
        return None


def _positive_delta_ok(n_new: int) -> bool:
    """Compare this run's positive count against the last published index —
    the guard against a classifier that got fine-tuned into flagging almost
    everything (or nothing). False when it moved by more than
    MERGE_MAX_POSITIVE_DELTA; True when it didn't, or can't be checked."""
    n_old = _published_positive_count(HF_DATASET_REPO)
    if n_old is None:
        return True

    if n_old == 0:
        delta = 0.0 if n_new == 0 else float("inf")
    else:
        delta = (n_new - n_old) / n_old

    print(f"positives: {n_old:,} -> {n_new:,} ({delta * 100:+.1f}%)")
    return abs(delta) <= MERGE_MAX_POSITIVE_DELTA


def incomplete_jobs(grid_tiles: dict) -> list:
    """Array task indexes whose results.csv lacks some of its grid tiles or
    holds retryable download failures — what slurm/resume_scan.sh resubmits.
    grid_tiles: job_idx -> set of that job's tile ids."""
    todo = []
    for job, expected in sorted(grid_tiles.items()):
        path = OUTPUTS_DIR / f"job_{job:04d}" / "results.csv"
        done = set()
        if path.exists():
            with open(path, newline="") as f:
                done = {r["tile_id"] for r in csv.DictReader(f) if not is_retryable(r)}
        if expected - done:
            todo.append(job)
    return todo


def _grid_tiles_by_job() -> dict:
    grid = load_grid(GRID_FILE)
    return {job: set(grid["tile_id"].iloc[job * TILES_PER_JOB:(job + 1) * TILES_PER_JOB])
            for job in range(n_jobs(GRID_FILE, TILES_PER_JOB))}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-errors", action="store_true",
                    help="publish even with missing tiles or more download errors than "
                         "MERGE_MAX_ERROR_FRACTION (they stay out of this round)")
    ap.add_argument("--accept-delta", action="store_true",
                    help="publish even though the positive count moved by more than "
                         "MERGE_MAX_POSITIVE_DELTA (after checking why)")
    ap.add_argument("--list-incomplete", action="store_true",
                    help="only print the comma-separated array task indexes with missing "
                         "or failed tiles, and exit")
    return ap.parse_args()


def main():
    args = parse_args()
    grid_tiles = _grid_tiles_by_job()
    if args.list_incomplete:
        print(",".join(str(j) for j in incomplete_jobs(grid_tiles)))
        return

    # ── Collect all job CSVs ─────────────────────────────────────────────
    csv_files = sorted(OUTPUTS_DIR.glob("job_*/results.csv"))
    if not csv_files:
        sys.exit("No result CSVs found. Run inference first.")

    df = pd.concat(
        [pd.read_csv(f).assign(job_idx=int(f.parent.name.split("_")[1])) for f in csv_files],
        ignore_index=True,
    )
    df.drop(columns="job_idx").to_csv(OUTPUTS_DIR / "all_results.csv", index=False)
    print(f"Total tiles processed : {len(df):,}")
    flagged = df["is_positive"].astype(str).str.lower().eq("true")
    # The export rule (src/results.py::is_detection) decides what this round
    # publishes, so the masks below and the positive-count guard use it too.
    detected = flagged & pd.Series(
        [is_detection(r.get("clf_prob"), patch_px_of(r, ROOT)) if f else False
         for r, f in zip(df.to_dict("records"), flagged)],
        index=df.index,
    )
    print(f"Flagged by classifier : {flagged.sum():,}")
    print(f"Detections (exported) : {detected.sum():,}  "
          f"(flagged, with a windthrow patch >= EXPORT_MIN_PATCH_PX)")

    # Same split as src/results.py::is_retryable: n_scenes == 0 is "no imagery
    # here" and final; anything else is an error a resubmit retries.
    failed   = ~df["download_ok"].astype(str).str.lower().eq("true")
    no_image = failed & (df["n_scenes"] == 0) if "n_scenes" in df else failed & False
    errors   = failed & ~no_image
    print(f"Download failures     : {failed.sum():,}  "
          f"({no_image.sum():,} without imagery, {errors.sum():,} download errors)")

    expected_ids = set().union(*grid_tiles.values())
    missing = expected_ids - set(df["tile_id"])
    with_imagery = len(df) - int(no_image.sum())
    error_frac = errors.sum() / with_imagery if with_imagery else 0.0
    problems = []
    if missing:
        problems.append(f"{len(missing):,} grid tiles have no result at all (an array task "
                        "died before finishing)")
    if error_frac > MERGE_MAX_ERROR_FRACTION:
        problems.append(f"{errors.sum():,} tiles ({error_frac:.1%} of those with imagery) failed to "
                        f"download — more than MERGE_MAX_ERROR_FRACTION={MERGE_MAX_ERROR_FRACTION:.0%}")
    elif errors.any():
        print(f"  ({errors.sum():,} download errors, {error_frac:.2%} — within "
              f"MERGE_MAX_ERROR_FRACTION; those tiles stay out of this round)")

    # ── Build VRT of all mask GeoTIFFs ───────────────────────────────────
    mask_paths = df.loc[detected & df["mask_path"].notna() & (df["mask_path"] != ""), "mask_path"].tolist()
    abs_masks  = [str(ROOT / p) for p in mask_paths]

    if abs_masks:
        # The VRT is a convenience for QGIS; nothing downstream reads it, so a
        # node without GDAL's command-line tools skips it instead of failing.
        vrt_path = str(OUTPUTS_DIR / "windthrow_amazon.vrt")
        if shutil.which("gdalbuildvrt"):
            subprocess.run(["gdalbuildvrt", "-resolution", "highest", vrt_path] + abs_masks, check=True)
            print(f"VRT mosaic written → {vrt_path}")
        else:
            print("  (gdalbuildvrt not on PATH — skipping the QGIS mosaic VRT)")

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

    blocked = False
    if problems:
        jobs = ",".join(str(j) for j in incomplete_jobs(grid_tiles))
        print("\n  !! " + "\n  !! ".join(problems))
        print(f"     unfinished array tasks: {jobs or '(none)'}")
        if args.allow_errors:
            print("     --allow-errors: publishing without those tiles")
        else:
            blocked = True
            print("     Finish them and publish: slurm/resume_scan.sh  "
                  "(or accept the gaps: slurm/resume_scan.sh --allow-errors)")

    if not _positive_delta_ok(int(detected.sum())):
        print(f"  !! the positive count moved by more than {MERGE_MAX_POSITIVE_DELTA:.0%} since the "
              "published inputs_index.csv — usually a drifted classifier (roll it back, LOOP.md), "
              "sometimes a broken download.")
        if args.accept_delta:
            print("     --accept-delta: publishing anyway")
        else:
            blocked = True
            print("     Once you know why and want this round published: "
                  "slurm/resume_scan.sh --accept-delta")

    if blocked:
        sys.exit("merge stopped: this round will not be exported (03 is cancelled).")


if __name__ == "__main__":
    main()

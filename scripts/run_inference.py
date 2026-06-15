"""
Main GPU inference script – called by the SLURM array job.

    python scripts/run_inference.py --job-idx $SLURM_ARRAY_TASK_ID

Outputs per job:
    outputs/job_{job_idx:04d}/results.csv         – metadata + positive flag per tile
    outputs/job_{job_idx:04d}/{tile_id}_mask.tif  – mask for positive tiles only
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    CLASSIFIER_PTH,
    GEE_KEY_PATH,
    GRID_FILE,
    OUTPUTS_DIR,
    SEGMENTER_PTH,
    TILES_PER_JOB,
)
from src.data import download_tile, init_gee, preprocess, save_mask
from src.models import load_classifier, load_segmenter
from src.pipeline import run_pipeline
from src.tile_grid import get_job_tiles


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-idx",  type=int, required=True,
                    help="SLURM_ARRAY_TASK_ID")
    ap.add_argument("--key-path", default=None,
                    help="Path to GEE service-account JSON "
                         f"(default: {GEE_KEY_PATH})")
    ap.add_argument("--date-from", default=None)
    ap.add_argument("--date-to",   default=None)
    return ap.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[job {args.job_idx}] device={device}")

    key_path = Path(args.key_path) if args.key_path else None
    init_gee(key_path=key_path)
    print(f"[job {args.job_idx}] GEE initialised")

    classifier = load_classifier(CLASSIFIER_PTH, device)
    segmenter  = load_segmenter(SEGMENTER_PTH,  device)
    print(f"[job {args.job_idx}] Models loaded (torch.compiled)")

    tiles   = get_job_tiles(GRID_FILE, args.job_idx, TILES_PER_JOB)
    out_dir = OUTPUTS_DIR / f"job_{args.job_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint: skip already-processed tiles ──────────────────────────
    done_csv = out_dir / "results.csv"
    done_ids: set = set()
    if done_csv.exists():
        with open(done_csv) as f:
            done_ids = {row["tile_id"] for row in csv.DictReader(f)}
        print(f"[job {args.job_idx}] Resuming – {len(done_ids)} tiles already done")

    csv_file = open(done_csv, "a", newline="")
    writer   = csv.writer(csv_file)
    if not done_ids:
        writer.writerow(["tile_id", "lon_min", "lat_min", "lon_max", "lat_max",
                         "is_positive", "mask_path", "download_ok"])

    dl_kwargs: dict = {}
    if args.date_from:
        dl_kwargs["date_from"] = args.date_from
    if args.date_to:
        dl_kwargs["date_to"] = args.date_to

    t_start = time.perf_counter()
    n_pos   = 0

    for _, row in tiles.iterrows():
        tile_id = row["tile_id"]
        if tile_id in done_ids:
            continue

        bbox = (row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"])

        # Download raw Landsat-8 bands (7, H, W)
        ms = download_tile(bbox, **dl_kwargs)
        if ms is None:
            writer.writerow([tile_id, *bbox, False, "", False])
            csv_file.flush()
            continue

        # Normalise with fixed training-time p5/p95 (same as gradio preprocess())
        ms_norm = preprocess(ms).squeeze(0).numpy()   # back to (7, H, W) numpy

        is_pos, seg_masks = run_pipeline([ms_norm], classifier, segmenter, device)

        mask_path = ""
        if is_pos[0] and seg_masks[0] is not None:
            n_pos += 1
            mp     = out_dir / f"{tile_id}_mask.tif"
            save_mask(seg_masks[0], bbox, mp)
            mask_path = str(mp.relative_to(ROOT))

        writer.writerow([tile_id, *bbox, is_pos[0], mask_path, True])
        csv_file.flush()

    csv_file.close()
    elapsed = time.perf_counter() - t_start
    n_done  = len(tiles) - len(done_ids)
    print(
        f"[job {args.job_idx}] Done – {n_done} tiles in {elapsed:.1f}s "
        f"({elapsed / max(n_done, 1) * 1000:.1f} ms/tile) | positives: {n_pos}"
    )


if __name__ == "__main__":
    main()

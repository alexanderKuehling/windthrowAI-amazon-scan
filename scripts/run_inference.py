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
from typing import List

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    BATCH_SIZE,
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
    ap.add_argument("--job-idx",   type=int, required=True)
    ap.add_argument("--key-path",  default=None,
                    help=f"GEE service-account JSON (default: {GEE_KEY_PATH})")
    ap.add_argument("--date-from", default=None)
    ap.add_argument("--date-to",   default=None)
    return ap.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"[job {args.job_idx}] device={device}  GPUs={n_gpus}")

    key_path = Path(args.key_path) if args.key_path else None
    init_gee(key_path=key_path)
    print(f"[job {args.job_idx}] GEE initialised")

    classifier = load_classifier(CLASSIFIER_PTH, device)
    segmenter  = load_segmenter(SEGMENTER_PTH,  device)
    print(f"[job {args.job_idx}] Models loaded")

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

    csv_file = open(done_csv, "a", newline="\n")
    writer   = csv.writer(csv_file)
    if not done_ids:
        writer.writerow(["tile_id", "lon_min", "lat_min", "lon_max", "lat_max",
                         "is_positive", "mask_path", "download_ok"])

    dl_kwargs: dict = {}
    if args.date_from: dl_kwargs["date_from"] = args.date_from
    if args.date_to:   dl_kwargs["date_to"]   = args.date_to

    # ── Download all tiles first, accumulate into batches for GPU ─────────
    # This way DataParallel always gets full batches (both GPUs stay busy).
    pending_ids:    List[str]            = []
    pending_bboxes: List[tuple]          = []
    pending_arrays: List[np.ndarray]     = []
    failed_rows:    List[list]           = []

    t_dl_start = time.perf_counter()
    for _, row in tiles.iterrows():
        tile_id = row["tile_id"]
        if tile_id in done_ids:
            continue
        bbox = (row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"])
        ms   = download_tile(bbox, **dl_kwargs)
        if ms is None:
            failed_rows.append([tile_id, *bbox, False, "", False])
            continue
        pending_ids.append(tile_id)
        pending_bboxes.append(bbox)
        pending_arrays.append(preprocess(ms).squeeze(0).numpy())  # (7, H, W)

    t_dl = time.perf_counter() - t_dl_start
    print(
        f"[job {args.job_idx}] Downloaded {len(pending_arrays)} tiles "
        f"({len(failed_rows)} failed) in {t_dl:.1f}s"
    )

    # Write download failures immediately
    for row in failed_rows:
        writer.writerow(row)
    csv_file.flush()

    # ── GPU inference in batches ──────────────────────────────────────────
    # Use 2× batch_size when 2 GPUs are present so each GPU handles batch_size items
    effective_bs = BATCH_SIZE * max(n_gpus, 1)

    t_gpu_start = time.perf_counter()
    is_positive, masks = run_pipeline(
        pending_arrays, classifier, segmenter, device, batch_size=effective_bs
    )
    t_gpu = time.perf_counter() - t_gpu_start
    print(
        f"[job {args.job_idx}] GPU inference done in {t_gpu:.1f}s "
        f"({t_gpu / max(len(pending_arrays), 1) * 1000:.2f} ms/tile)"
    )

    # ── Write results ─────────────────────────────────────────────────────
    n_pos = 0
    for tile_id, bbox, pos, mask in zip(pending_ids, pending_bboxes, is_positive, masks):
        mask_path = ""
        if pos and mask is not None:
            n_pos += 1
            mp        = out_dir / f"{tile_id}_mask.tif"
            save_mask(mask, bbox, mp)
            mask_path = str(mp.relative_to(ROOT))
        writer.writerow([tile_id, *bbox, pos, mask_path, True])

    csv_file.flush()
    csv_file.close()

    total = len(pending_arrays)
    print(f"[job {args.job_idx}] positives: {n_pos}/{total}")


if __name__ == "__main__":
    main()

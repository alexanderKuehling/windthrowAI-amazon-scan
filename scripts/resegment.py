"""
Re-run segmentation only, with a different segmenter checkpoint, on tiles
already flagged positive by the classifier. Skips re-classification and
re-downloads only the (small) positive subset.

    python scripts/resegment.py --job-idx 0 --segmenter-path models/segmenter_v2.pth

Outputs:
    outputs_resegment/job_{job_idx:04d}/results.csv
    outputs_resegment/job_{job_idx:04d}/{tile_id}_mask.tif
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import BATCH_SIZE, OUTPUTS_DIR, SEGMENTER_BACKBONE, TILES_PER_JOB
from src.data import download_tile, init_gee, preprocess, save_mask
from src.models import load_segmenter
from src.pipeline import segment_batch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-idx",        type=int, required=True)
    ap.add_argument("--segmenter-path", type=Path, required=True)
    ap.add_argument("--backbone",       default=SEGMENTER_BACKBONE,
                     help="smp encoder name matching the checkpoint, e.g. resnet34, vgg19_bn")
    ap.add_argument("--chunk-size",     type=int, default=TILES_PER_JOB)
    ap.add_argument("--results-csv",    type=Path, default=OUTPUTS_DIR / "all_results.csv")
    ap.add_argument("--out-dir",        type=Path, default=ROOT / "outputs_resegment")
    ap.add_argument("--key-path",       default=None)
    return ap.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"[resegment {args.job_idx}] device={device}  GPUs={n_gpus}")

    key_path = Path(args.key_path) if args.key_path else None
    init_gee(key_path=key_path)

    segmenter = load_segmenter(args.segmenter_path, device, encoder_name=args.backbone)
    print(f"[resegment {args.job_idx}] segmenter loaded: {args.segmenter_path.name}")

    df  = pd.read_csv(args.results_csv)
    pos = df[df["is_positive"] == True].reset_index(drop=True)

    start = args.job_idx * args.chunk_size
    end   = start + args.chunk_size
    chunk = pos.iloc[start:end]
    print(
        f"[resegment {args.job_idx}] {len(chunk)} positive tiles in this chunk "
        f"(of {len(pos)} total)"
    )

    out_dir = args.out_dir / f"job_{args.job_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    done_csv = out_dir / "results.csv"
    done_ids: set = set()
    if done_csv.exists():
        with open(done_csv) as f:
            done_ids = {row["tile_id"] for row in csv.DictReader(f)}
        print(f"[resegment {args.job_idx}] resuming - {len(done_ids)} tiles already done")

    csv_file = open(done_csv, "a", newline="\n")
    writer   = csv.writer(csv_file)
    if not done_ids:
        writer.writerow(["tile_id", "lon_min", "lat_min", "lon_max", "lat_max",
                          "is_positive", "mask_path"])

    pending_ids, pending_bboxes, pending_arrays = [], [], []
    failed_ids = []

    t0 = time.perf_counter()
    for _, row in chunk.iterrows():
        tile_id = row["tile_id"]
        if tile_id in done_ids:
            continue
        bbox = (row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"])
        ms   = download_tile(bbox)
        if ms is None:
            failed_ids.append(tile_id)
            continue
        pending_ids.append(tile_id)
        pending_bboxes.append(bbox)
        pending_arrays.append(preprocess(ms).squeeze(0).numpy())

    print(
        f"[resegment {args.job_idx}] downloaded {len(pending_arrays)} tiles "
        f"({len(failed_ids)} failed) in {time.perf_counter() - t0:.1f}s"
    )

    effective_bs = BATCH_SIZE * max(n_gpus, 1)
    n_pos = 0
    for start_i in range(0, len(pending_arrays), effective_bs):
        batch_ids    = pending_ids[start_i : start_i + effective_bs]
        batch_bboxes = pending_bboxes[start_i : start_i + effective_bs]
        batch_np     = np.stack(pending_arrays[start_i : start_i + effective_bs], axis=0)
        batch_t      = torch.from_numpy(batch_np).to(device, non_blocking=True)
        masks        = segment_batch(batch_t, segmenter)

        for tile_id, bbox, mask in zip(batch_ids, batch_bboxes, masks):
            is_pos    = bool(mask.sum() > 0)
            mask_path = ""
            if is_pos:
                n_pos += 1
                mp = out_dir / f"{tile_id}_mask.tif"
                save_mask(mask, bbox, mp)
                mask_path = str(mp.relative_to(ROOT))
            writer.writerow([tile_id, *bbox, is_pos, mask_path])

    csv_file.flush()
    csv_file.close()
    print(
        f"[resegment {args.job_idx}] positives (new segmenter): "
        f"{n_pos}/{len(pending_arrays)}"
    )


if __name__ == "__main__":
    main()

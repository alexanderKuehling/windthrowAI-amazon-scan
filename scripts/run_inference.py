"""
Main GPU inference script – called by the SLURM array job.

    python scripts/run_inference.py --job-idx $SLURM_ARRAY_TASK_ID

Checkpoints and the segmenter encoder are selectable (same as scripts/resegment.py),
so a checkpoint validated there can be run end-to-end here:

    python scripts/run_inference.py --job-idx 0 \
        --segmenter-path models/segmenter_v2.pth --backbone resnet34

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
    SEGMENTER_BACKBONE,
    SEGMENTER_PTH,
    TILES_PER_JOB,
)
from src.data import download_tile, init_gee, preprocess, save_input, save_mask
from src.models import load_classifier, load_segmenter
from src.pipeline import classify_batch, segment_batch
from src.tile_grid import get_job_tiles


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-idx",         type=int, required=True)
    ap.add_argument("--classifier-path", type=Path, default=CLASSIFIER_PTH)
    ap.add_argument("--segmenter-path",  type=Path, default=SEGMENTER_PTH)
    ap.add_argument("--backbone",        default=SEGMENTER_BACKBONE,
                    help="smp encoder name matching the segmenter checkpoint, "
                         "e.g. resnet34, vgg19_bn")
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

    classifier = load_classifier(args.classifier_path, device)
    segmenter  = load_segmenter(args.segmenter_path, device, encoder_name=args.backbone)
    print(
        f"[job {args.job_idx}] Models loaded – "
        f"classifier: {args.classifier_path.name}, "
        f"segmenter: {args.segmenter_path.name} ({args.backbone})"
    )

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
                         "is_positive", "mask_path", "input_path", "download_ok"])

    dl_kwargs: dict = {}
    if args.date_from: dl_kwargs["date_from"] = args.date_from
    if args.date_to:   dl_kwargs["date_to"]   = args.date_to

    # ── Download all tiles first, accumulate into batches for GPU ─────────
    # This way DataParallel always gets full batches (both GPUs stay busy).
    pending_ids:    List[str]            = []
    pending_bboxes: List[tuple]          = []
    pending_arrays: List[np.ndarray]     = []
    pending_raw:    List[np.ndarray]     = []
    failed_rows:    List[list]           = []

    t_dl_start = time.perf_counter()
    for _, row in tiles.iterrows():
        tile_id = row["tile_id"]
        if tile_id in done_ids:
            continue
        bbox = (row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"])
        ms   = download_tile(bbox, **dl_kwargs)
        if ms is None:
            failed_rows.append([tile_id, *bbox, False, "", "", False])
            continue
        pending_ids.append(tile_id)
        pending_bboxes.append(bbox)
        pending_raw.append(ms)                                    # (7, H, W) raw
        pending_arrays.append(preprocess(ms).squeeze(0).numpy())  # (7, H, W) normalised

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
    n_pos = 0
    for start_i in range(0, len(pending_arrays), effective_bs):
        batch_ids    = pending_ids[start_i : start_i + effective_bs]
        batch_bboxes = pending_bboxes[start_i : start_i + effective_bs]
        batch_raw    = pending_raw[start_i : start_i + effective_bs]
        batch_np     = np.stack(pending_arrays[start_i : start_i + effective_bs], axis=0)
        batch_t      = torch.from_numpy(batch_np).to(device, non_blocking=True)

        # Stage 1: classify the whole batch
        preds = classify_batch(batch_t, classifier)          # bool (B,)

        # Stage 2: segment only the tiles the classifier gated through
        pos_local = [i for i, p in enumerate(preds) if p]
        masks: dict = {}
        if pos_local:
            idx       = torch.as_tensor(pos_local, device=batch_t.device)
            seg_masks = segment_batch(batch_t.index_select(0, idx), segmenter)
            masks     = {li: seg_masks[k] for k, li in enumerate(pos_local)}

        # Write this batch out immediately – a crash keeps everything so far
        for i, (tile_id, bbox, raw) in enumerate(zip(batch_ids, batch_bboxes, batch_raw)):
            pos        = bool(preds[i])
            mask       = masks.get(i)
            mask_path  = ""
            input_path = ""
            if pos and mask is not None:
                n_pos += 1
                mp        = out_dir / f"{tile_id}_mask.tif"
                save_mask(mask, bbox, mp)
                mask_path = str(mp.relative_to(ROOT))

                ip = out_dir / f"{tile_id}_input.tif"
                save_input(raw, bbox, ip)
                input_path = str(ip.relative_to(ROOT))
            writer.writerow([tile_id, *bbox, pos, mask_path, input_path, True])
        csv_file.flush()

    t_gpu = time.perf_counter() - t_gpu_start
    print(
        f"[job {args.job_idx}] GPU inference done in {t_gpu:.1f}s "
        f"({t_gpu / max(len(pending_arrays), 1) * 1000:.2f} ms/tile)"
    )

    csv_file.close()

    total = len(pending_arrays)
    print(f"[job {args.job_idx}] positives: {n_pos}/{total}")


if __name__ == "__main__":
    main()

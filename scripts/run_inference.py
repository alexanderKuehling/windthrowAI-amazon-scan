"""
Main GPU inference script – called by the SLURM array job.

    python scripts/run_inference.py --job-idx $SLURM_ARRAY_TASK_ID

Checkpoints and the segmenter encoder are selectable (same as scripts/resegment.py),
so a checkpoint validated there can be run end-to-end here:

    python scripts/run_inference.py --job-idx 0 \
        --segmenter-path models/segmenter_v2.pth --backbone resnet34

Outputs per job:
    outputs/job_{job_idx:04d}/results.csv         – metadata + scores per tile
    outputs/job_{job_idx:04d}/run_meta.json       – checkpoints, thresholds, dates
    outputs/job_{job_idx:04d}/{tile_id}_mask.tif  – mask for positive tiles only
    outputs/job_{job_idx:04d}/{tile_id}_input.tif – model input for positive tiles
    outputs/job_{job_idx:04d}/{tile_id}_prob.tif  – segmenter probability map

results.csv records the model's *scores*, not just its decisions: `clf_prob` for
every downloaded tile (including negatives) and the segmenter probability map for
every positive one. Both thresholds therefore become post-processing choices —
re-thresholding needs no GPU pass and no second GEE download.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    BATCH_SIZE,
    CLASSIFIER_PTH,
    CLASSIFIER_THRESHOLD,
    CLOUD_COVER_MAX,
    DATE_FROM,
    DATE_TO,
    GEE_KEY_PATH,
    GRID_FILE,
    OUTPUTS_DIR,
    RESOLUTION_M,
    SEGMENTER_BACKBONE,
    SEGMENTER_PTH,
    SEGMENTER_THRESHOLD,
    TILE_SIZE_PX,
    TILES_PER_JOB,
)
from src.data import download_tile_with_meta, init_gee, preprocess, save_input, save_mask, save_prob_map
from src.models import load_classifier, load_segmenter
from src.pipeline import classify_probs, segment_probs
from src.results import base_row, fill_segmentation, open_results_csv
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
    ap.add_argument("--clf-threshold", type=float, default=CLASSIFIER_THRESHOLD,
                    help="gate for stage 1; clf_prob is recorded regardless")
    ap.add_argument("--seg-threshold", type=float, default=SEGMENTER_THRESHOLD,
                    help="binarisation of the segmenter output; the probability "
                         "map is saved regardless")
    ap.add_argument("--no-prob-maps", action="store_true",
                    help="skip {tile_id}_prob.tif (loses post-hoc re-thresholding)")
    return ap.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — provenance only
        return ""


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

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── Checkpoint: skip already-processed tiles ──────────────────────────
    csv_file, writer, done_ids = open_results_csv(out_dir / "results.csv")
    if done_ids:
        print(f"[job {args.job_idx}] Resuming – {len(done_ids)} tiles already done")

    dl_kwargs: dict = {}
    if args.date_from: dl_kwargs["date_from"] = args.date_from
    if args.date_to:   dl_kwargs["date_to"]   = args.date_to

    # ── Download all tiles first, accumulate into batches for GPU ─────────
    # This way DataParallel always gets full batches (both GPUs stay busy).
    pending_ids:    List[str]            = []
    pending_bboxes: List[tuple]          = []
    pending_arrays: List[np.ndarray]     = []
    pending_raw:    List[np.ndarray]     = []
    pending_meta:   List[dict]           = []
    failed_rows:    List[dict]           = []

    t_dl_start = time.perf_counter()
    for _, row in tiles.iterrows():
        tile_id = row["tile_id"]
        if tile_id in done_ids:
            continue
        bbox = (row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"])
        ms, meta = download_tile_with_meta(bbox, **dl_kwargs)
        if ms is None:
            failed_rows.append(base_row(tile_id, bbox, meta, download_ok=False))
            continue
        pending_ids.append(tile_id)
        pending_bboxes.append(bbox)
        pending_meta.append(meta)
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
        sl           = slice(start_i, start_i + effective_bs)
        batch_ids    = pending_ids[sl]
        batch_bboxes = pending_bboxes[sl]
        batch_raw    = pending_raw[sl]
        batch_meta   = pending_meta[sl]
        batch_np     = np.stack(pending_arrays[sl], axis=0)
        batch_t      = torch.from_numpy(batch_np).to(device, non_blocking=True)

        # Stage 1: classify the whole batch — the probability is kept, not just
        # the gate decision, so the threshold can be revisited without a re-scan.
        clf_probs = classify_probs(batch_t, classifier)      # float32 (B,)
        preds     = clf_probs >= args.clf_threshold

        # Stage 2: segment only the tiles the classifier gated through
        pos_local = [i for i, p in enumerate(preds) if p]
        seg: dict = {}
        if pos_local:
            idx  = torch.as_tensor(pos_local, device=batch_t.device)
            maps = segment_probs(batch_t.index_select(0, idx), segmenter)  # (P, H, W)
            seg  = {li: maps[k] for k, li in enumerate(pos_local)}

        # Write this batch out immediately – a crash keeps everything so far
        for i, (tile_id, bbox, raw, meta) in enumerate(
            zip(batch_ids, batch_bboxes, batch_raw, batch_meta)
        ):
            record = base_row(tile_id, bbox, meta, download_ok=True)
            record["clf_prob"] = round(float(clf_probs[i]), 6)

            pos  = bool(preds[i])
            prob = seg.get(i)
            record["is_positive"] = pos
            if pos and prob is not None:
                n_pos += 1
                mask = (prob >= args.seg_threshold).astype(np.uint8)

                mp = out_dir / f"{tile_id}_mask.tif"
                save_mask(mask, bbox, mp)
                record["mask_path"] = str(mp.relative_to(ROOT))

                ip = out_dir / f"{tile_id}_input.tif"
                save_input(raw, bbox, ip)
                record["input_path"] = str(ip.relative_to(ROOT))

                if not args.no_prob_maps:
                    pp = out_dir / f"{tile_id}_prob.tif"
                    save_prob_map(prob, bbox, pp)
                    record["prob_path"] = str(pp.relative_to(ROOT))

                fill_segmentation(record, prob, mask)
            writer.writerow(record)
        csv_file.flush()

    t_gpu = time.perf_counter() - t_gpu_start
    print(
        f"[job {args.job_idx}] GPU inference done in {t_gpu:.1f}s "
        f"({t_gpu / max(len(pending_arrays), 1) * 1000:.2f} ms/tile)"
    )

    csv_file.close()

    total = len(pending_arrays)
    print(f"[job {args.job_idx}] positives: {n_pos}/{total}")

    # ── Run provenance ────────────────────────────────────────────────────
    # Which checkpoints and thresholds produced this job's rows, recorded next to
    # them: results.csv alone cannot tell two runs apart after the fact.
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps({
        "job_idx":              args.job_idx,
        "started_at":           started_at,
        "finished_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classifier_path":      str(args.classifier_path),
        "segmenter_path":       str(args.segmenter_path),
        "segmenter_backbone":   args.backbone,
        "classifier_threshold": args.clf_threshold,
        "segmenter_threshold":  args.seg_threshold,
        "date_from":            args.date_from or DATE_FROM,
        "date_to":              args.date_to   or DATE_TO,
        "cloud_cover_max":      CLOUD_COVER_MAX,
        "tile_size_px":         TILE_SIZE_PX,
        "resolution_m":         RESOLUTION_M,
        "prob_maps":            not args.no_prob_maps,
        "n_tiles_processed":    total,
        "n_download_failed":    len(failed_rows),
        "n_positive":           n_pos,
        "git_commit":           _git_commit(),
        "host":                 platform.node(),
    }, indent=2))
    print(f"[job {args.job_idx}] wrote {meta_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

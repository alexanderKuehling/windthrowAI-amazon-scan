"""
Re-run segmentation only, with a different segmenter checkpoint, on tiles
already flagged positive by the classifier. Skips re-classification and
re-downloads only the (small) positive subset.

    python scripts/resegment.py --job-idx 0 --segmenter-path models/segmenter_v2.pth

Outputs:
    outputs_resegment/job_{job_idx:04d}/results.csv
    outputs_resegment/job_{job_idx:04d}/{tile_id}_mask.tif
    outputs_resegment/job_{job_idx:04d}/{tile_id}_prob.tif

Mirrors the column set run_inference.py writes, so the export reads either
directory the same way. `clf_prob` is carried over from the source results CSV —
this step never re-runs the classifier.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    BATCH_SIZE,
    OUTPUTS_DIR,
    SEGMENTER_BACKBONE,
    SEGMENTER_THRESHOLD,
    TILES_PER_JOB,
)
from src.data import (
    download_tile_with_meta,
    init_gee,
    preprocess,
    save_input,
    save_mask,
    save_prob_map,
)
from src.models import load_segmenter
from src.pipeline import segment_probs
from src.results import base_row, fill_segmentation, open_results_csv


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
    ap.add_argument("--seg-threshold",  type=float, default=SEGMENTER_THRESHOLD,
                     help="binarisation of the segmenter output; the probability "
                          "map is saved regardless")
    ap.add_argument("--no-prob-maps",   action="store_true",
                     help="skip {tile_id}_prob.tif (loses post-hoc re-thresholding)")
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

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    csv_file, writer, done_ids = open_results_csv(out_dir / "results.csv")
    if done_ids:
        print(f"[resegment {args.job_idx}] resuming - {len(done_ids)} tiles already done")

    # Stage-1 scores are not recomputed here — carry them over from the source run
    # so a re-segmented tile keeps the classifier confidence it was selected on.
    # Empty for tiles scanned before clf_prob was recorded (NaN, not the string "nan").
    clf_probs = {}
    if "clf_prob" in df.columns:
        clf_probs = {t: p for t, p in zip(df["tile_id"], df["clf_prob"]) if pd.notna(p)}

    pending_ids, pending_bboxes, pending_arrays, pending_raw, pending_meta = [], [], [], [], []
    failed_ids = []

    t0 = time.perf_counter()
    for _, row in chunk.iterrows():
        tile_id = row["tile_id"]
        if tile_id in done_ids:
            continue
        bbox = (row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"])
        ms, meta = download_tile_with_meta(bbox)
        if ms is None:
            failed_ids.append(tile_id)
            writer.writerow(base_row(tile_id, bbox, meta, download_ok=False))
            continue
        pending_ids.append(tile_id)
        pending_bboxes.append(bbox)
        pending_meta.append(meta)
        pending_raw.append(ms)
        pending_arrays.append(preprocess(ms).squeeze(0).numpy())
    csv_file.flush()

    print(
        f"[resegment {args.job_idx}] downloaded {len(pending_arrays)} tiles "
        f"({len(failed_ids)} failed) in {time.perf_counter() - t0:.1f}s"
    )

    effective_bs = BATCH_SIZE * max(n_gpus, 1)
    n_pos = 0
    for start_i in range(0, len(pending_arrays), effective_bs):
        sl           = slice(start_i, start_i + effective_bs)
        batch_ids    = pending_ids[sl]
        batch_bboxes = pending_bboxes[sl]
        batch_raw    = pending_raw[sl]
        batch_meta   = pending_meta[sl]
        batch_np     = np.stack(pending_arrays[sl], axis=0)
        batch_t      = torch.from_numpy(batch_np).to(device, non_blocking=True)
        probs        = segment_probs(batch_t, segmenter)   # (B, H, W) float

        for tile_id, bbox, prob, raw, meta in zip(
            batch_ids, batch_bboxes, probs, batch_raw, batch_meta
        ):
            mask   = (prob >= args.seg_threshold).astype(np.uint8)
            is_pos = bool(mask.sum() > 0)

            record = base_row(tile_id, bbox, meta, download_ok=True)
            record["is_positive"] = is_pos
            record["clf_prob"]    = clf_probs.get(tile_id, "")
            fill_segmentation(record, prob, mask)   # recorded for rejects too:
            # a tile the new segmenter drops is still evidence about the model
            if is_pos:
                n_pos += 1
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
            writer.writerow(record)
        csv_file.flush()

    csv_file.close()
    print(
        f"[resegment {args.job_idx}] positives (new segmenter): "
        f"{n_pos}/{len(pending_arrays)}"
    )

    (out_dir / "run_meta.json").write_text(json.dumps({
        "job_idx":             args.job_idx,
        "started_at":          started_at,
        "finished_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segmenter_path":      str(args.segmenter_path),
        "segmenter_backbone":  args.backbone,
        "segmenter_threshold": args.seg_threshold,
        "source_results_csv":  str(args.results_csv),
        "chunk_size":          args.chunk_size,
        "prob_maps":           not args.no_prob_maps,
        "n_tiles_processed":   len(pending_arrays),
        "n_download_failed":   len(failed_ids),
        "n_positive":          n_pos,
    }, indent=2))


if __name__ == "__main__":
    main()

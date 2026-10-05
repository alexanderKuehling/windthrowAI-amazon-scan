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
    outputs/job_{job_idx:04d}/{tile_id}_input.tif – model input for positive tiles,
                                                    and for rejected spot-check tiles
    outputs/job_{job_idx:04d}/{tile_id}_prob.tif  – segmenter probability map

A small fixed sample of the tiles the classifier *rejects* keeps its input too
(is_control=True, see src/results.py::is_control_sample): the export publishes
them as review spot checks, the only way reviewers ever see what the model misses.

results.csv records the model's *scores*, not just its decisions: `clf_prob` for
every downloaded tile (including negatives) and the segmenter probability map for
every positive one. Both thresholds therefore become post-processing choices —
re-thresholding needs no GPU pass and no second GEE download.

Tiles are processed DOWNLOAD_CHUNK at a time — download, classify, write — so a
task that hits its walltime keeps every finished chunk, and its resubmit
downloads only the rest. --source picks the imagery: GEE or Microsoft Planetary
Computer (src/sources.py, default config.DATA_SOURCE).
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    NBR_ENCODING,
    BATCH_SIZE,
    CLASSIFIER_THRESHOLD,
    CLOUD_COVER_MAX,
    DATE_FROM,
    DATE_TO,
    DATA_SOURCE,
    DOWNLOAD_CHUNK,
    GEE_KEY_PATH,
    GRID_FILE,
    MODEL_VERSIONS_FILE,
    OUTPUTS_DIR,
    RESOLUTION_M,
    SEGMENTER_THRESHOLD,
    TILE_SIZE_PX,
    TILES_PER_JOB,
)
from src.data import preprocess, save_input, save_mask, save_prob_map
from src.hub import download_checkpoint
from src.models import load_classifier, load_segmenter
from src.pipeline import classify_probs, segment_probs
from src.results import base_row, fill_segmentation, is_control_sample, is_retryable, open_results_csv
from src.sources import SOURCES, default_workers, download_in_chunks, get_downloader
from src.tile_grid import get_job_tiles


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-idx",         type=int, required=True)
    ap.add_argument("--classifier-path", type=Path, default=None,
                    help="local checkpoint override; default: pinned HF pull "
                         "via model_versions.json (see resolve_model_versions.py)")
    ap.add_argument("--segmenter-path",  type=Path, default=None,
                    help="local checkpoint override; default: pinned HF pull "
                         "via model_versions.json (see resolve_model_versions.py)")
    ap.add_argument("--backbone",        default=None,
                    help="smp encoder name matching the segmenter checkpoint, "
                         "e.g. resnet34, vgg19_bn — default: auto-detect from "
                         "the checkpoint's embedded backbone field")
    ap.add_argument("--key-path",  default=None,
                    help=f"GEE service-account JSON (default: {GEE_KEY_PATH})")
    ap.add_argument("--date-from", default=None)
    ap.add_argument("--date-to",   default=None)
    ap.add_argument("--source", choices=SOURCES, default=DATA_SOURCE,
                    help="imagery source: gee (Earth Engine) or pc (Microsoft "
                         "Planetary Computer); default config.DATA_SOURCE")
    ap.add_argument("--workers", type=int, default=None,
                    help="tiles downloaded concurrently in this task (default: "
                         "GEE_WORKERS for gee — times the array's %%N it must stay "
                         "under the project's 40 — or PC_WORKERS for pc)")
    ap.add_argument("--chunk-tiles", type=int, default=DOWNLOAD_CHUNK,
                    help="tiles downloaded, classified and written per round")
    ap.add_argument("--clf-threshold", type=float, default=CLASSIFIER_THRESHOLD,
                    help="gate for stage 1; clf_prob is recorded regardless")
    ap.add_argument("--seg-threshold", type=float, default=SEGMENTER_THRESHOLD,
                    help="binarisation of the segmenter output; the probability "
                         "map is saved regardless")
    ap.add_argument("--no-prob-maps", action="store_true",
                    help="skip {tile_id}_prob.tif (loses post-hoc re-thresholding)")
    return ap.parse_args()


def _resolve_checkpoint_paths(args):
    """
    Fill in --classifier-path/--segmenter-path from the pinned HF versions file
    when either is not given explicitly on the command line. An explicit path
    always wins (today's manual-override workflow for testing an unpublished
    candidate checkpoint, unchanged).

    Returns (classifier_path, segmenter_path, revisions) where `revisions` is
    {"classifier_revision": ..., "segmenter_revision": ...} for provenance —
    None for whichever side used an explicit local override.
    """
    if args.classifier_path is not None and args.segmenter_path is not None:
        return args.classifier_path, args.segmenter_path, {
            "classifier_revision": None,
            "segmenter_revision":  None,
        }

    if not MODEL_VERSIONS_FILE.exists():
        sys.exit(
            "No --classifier-path/--segmenter-path given and "
            f"{MODEL_VERSIONS_FILE.name} is missing. Run "
            "scripts/resolve_model_versions.py first (or use "
            "slurm/run_full_scan.sh, which does this automatically)."
        )
    versions = json.loads(MODEL_VERSIONS_FILE.read_text())

    classifier_path = args.classifier_path or download_checkpoint(
        versions["classifier"]["repo_id"], versions["classifier"]["filename"],
        versions["classifier"]["revision"], local_files_only=True,
    )
    segmenter_path = args.segmenter_path or download_checkpoint(
        versions["segmenter"]["repo_id"], versions["segmenter"]["filename"],
        versions["segmenter"]["revision"], local_files_only=True,
    )
    return classifier_path, segmenter_path, {
        "classifier_revision": None if args.classifier_path else versions["classifier"]["revision"],
        "segmenter_revision":  None if args.segmenter_path  else versions["segmenter"]["revision"],
    }


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

    classifier_path, segmenter_path, revisions = _resolve_checkpoint_paths(args)

    key_path = Path(args.key_path) if args.key_path else None
    download = get_downloader(args.source, key_path=key_path)
    workers  = args.workers or default_workers(args.source)
    print(f"[job {args.job_idx}] imagery source: {args.source} ({workers} tiles in flight)")

    classifier = load_classifier(classifier_path, device)
    segmenter, segmenter_backbone = load_segmenter(
        segmenter_path, device, encoder_name=args.backbone
    )
    print(
        f"[job {args.job_idx}] Models loaded – "
        f"classifier: {classifier_path.name}, "
        f"segmenter: {segmenter_path.name} ({segmenter_backbone})"
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

    todo = [
        (row["tile_id"], (row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"]))
        for _, row in tiles.iterrows() if row["tile_id"] not in done_ids
    ]

    # Use 2× batch_size when 2 GPUs are present so each GPU handles batch_size items
    effective_bs = BATCH_SIZE * max(n_gpus, 1)

    def infer_and_write(items) -> tuple:
        """Classify, segment and write one chunk's downloaded tiles; returns
        (n_positive, n_control). Every batch is flushed to results.csv at once."""
        n_pos_chunk, n_control_chunk = 0, 0
        for start_i in range(0, len(items), effective_bs):
            batch   = items[start_i : start_i + effective_bs]
            batch_t = torch.from_numpy(
                np.stack([preprocess(ms).squeeze(0).numpy() for _, _, ms, _ in batch], axis=0)
            ).to(device, non_blocking=True)

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

            for i, (tile_id, bbox, raw, meta) in enumerate(batch):
                record = base_row(tile_id, bbox, meta, download_ok=True)
                record["clf_prob"] = round(float(clf_probs[i]), 6)

                pos  = bool(preds[i])
                prob = seg.get(i)
                record["is_positive"] = pos
                if pos and prob is not None:
                    n_pos_chunk += 1
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
                elif not pos and is_control_sample(tile_id, meta.get("valid_frac")):
                    n_control_chunk += 1
                    ip = out_dir / f"{tile_id}_input.tif"
                    save_input(raw, bbox, ip)
                    record["input_path"] = str(ip.relative_to(ROOT))
                    record["is_control"] = True
                writer.writerow(record)
            csv_file.flush()
        return n_pos_chunk, n_control_chunk

    # ── Download, infer, write — one chunk at a time ──────────────────────
    t_start = time.perf_counter()
    t_dl, t_gpu = 0.0, 0.0
    total, n_pos, n_control, n_failed, n_errors = 0, 0, 0, 0, 0
    t_mark = time.perf_counter()
    for chunk in download_in_chunks(todo, download, workers, args.chunk_tiles, **dl_kwargs):
        ok, failed = [], []
        for tile_id, bbox, ms, meta in chunk:
            if ms is None:
                failed.append(base_row(tile_id, bbox, meta, download_ok=False))
            else:
                ok.append((tile_id, bbox, ms, meta))
        t_dl += time.perf_counter() - t_mark

        for row in failed:   # failures first: a crash in the GPU part keeps them
            writer.writerow(row)
        csv_file.flush()

        t_mark = time.perf_counter()
        p, c = infer_and_write(ok)
        t_gpu += time.perf_counter() - t_mark

        total     += len(ok)
        n_pos     += p
        n_control += c
        n_failed  += len(failed)
        n_errors  += sum(1 for r in failed if is_retryable(r))
        print(f"[job {args.job_idx}]   {total + n_failed}/{len(todo)} tiles "
              f"({time.perf_counter() - t_start:.0f}s) — {n_pos} positive, {n_errors} failed so far")
        t_mark = time.perf_counter()

    csv_file.close()
    print(
        f"[job {args.job_idx}] Downloaded {total} tiles in {t_dl:.1f}s — "
        f"{n_failed - n_errors} without imagery, {n_errors} failed "
        f"(retried when this task is resubmitted); GPU {t_gpu:.1f}s "
        f"({t_gpu / max(total, 1) * 1000:.2f} ms/tile)"
    )
    print(f"[job {args.job_idx}] positives: {n_pos}/{total}  spot-check tiles: {n_control}")

    # ── Run provenance ────────────────────────────────────────────────────
    # Which checkpoints and thresholds produced this job's rows, recorded next to
    # them: results.csv alone cannot tell two runs apart after the fact. The
    # counts cover this invocation only; a resumed task's earlier rows are in
    # results.csv.
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps({
        "job_idx":              args.job_idx,
        "started_at":           started_at,
        "finished_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classifier_path":      str(classifier_path),
        "segmenter_path":       str(segmenter_path),
        "segmenter_backbone":   segmenter_backbone,
        "classifier_revision":  revisions["classifier_revision"],
        "segmenter_revision":   revisions["segmenter_revision"],
        "classifier_threshold": args.clf_threshold,
        "segmenter_threshold":  args.seg_threshold,
        "date_from":            args.date_from or DATE_FROM,
        "date_to":              args.date_to   or DATE_TO,
        "cloud_cover_max":      CLOUD_COVER_MAX,
        "tile_size_px":         TILE_SIZE_PX,
        "resolution_m":         RESOLUTION_M,
        "data_source":          args.source,
        "nbr_encoding":         NBR_ENCODING,
        "prob_maps":            not args.no_prob_maps,
        "n_tiles_processed":    total,
        "n_download_failed":    n_failed,
        "n_download_retryable": n_errors,
        "workers":              workers,
        "n_positive":           n_pos,
        "n_control":            n_control,
        "git_commit":           _git_commit(),
        "host":                 platform.node(),
    }, indent=2))
    print(f"[job {args.job_idx}] wrote {meta_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

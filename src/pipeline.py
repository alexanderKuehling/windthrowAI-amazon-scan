"""Two-stage gated inference: classify → gate → segment."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from .config import (
    BATCH_SIZE,
    CLASSIFIER_THRESHOLD,
    SEGMENTER_THRESHOLD,
)
from .models import WindthrowClassifier, WindthrowSegmenter


@torch.no_grad()
def classify_batch(
    tensors:    torch.Tensor,           # (B, 7, H, W) normalised, on device
    classifier: WindthrowClassifier,
    threshold:  float = CLASSIFIER_THRESHOLD,
    device:     torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Return bool array (B,) – True = classifier predicts windthrow."""
    with autocast(device_type="cuda", dtype=torch.float16):
        logits = classifier(tensors)    # (B,)
    probs = torch.sigmoid(logits.float()).cpu().numpy()
    return probs >= threshold


@torch.no_grad()
def segment_batch(
    tensors:   torch.Tensor,            # (B, 7, H, W) normalised, on device
    segmenter: WindthrowSegmenter,
    threshold: float = SEGMENTER_THRESHOLD,
) -> np.ndarray:
    """Return uint8 array (B, H, W) – binary segmentation masks."""
    with autocast(device_type="cuda", dtype=torch.float16):
        logits = segmenter(tensors)     # (B, 1, H, W)
    probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()  # (B, H, W)
    return (probs >= threshold).astype(np.uint8)


def run_pipeline(
    tile_arrays: List[np.ndarray],      # each: float32 (7, H, W), raw (not normalised)
    classifier:  WindthrowClassifier,
    segmenter:   WindthrowSegmenter,
    device:      torch.device = torch.device("cuda"),
    batch_size:  int = BATCH_SIZE,
) -> Tuple[List[bool], List[np.ndarray]]:
    """
    Run the gated two-stage pipeline over a list of pre-normalised tile arrays.

    Returns
    -------
    is_positive : List[bool]  – True if classifier flagged the tile
    masks       : List[ndarray | None]  – uint8 (H, W) mask for positives, else None
    """
    n             = len(tile_arrays)
    is_positive   = [False] * n
    masks: List   = [None]  * n

    # ── Stage 1: classify all tiles in batches ───────────────────────────
    for start in range(0, n, batch_size):
        batch_np = np.stack(tile_arrays[start : start + batch_size], axis=0)
        batch_t  = torch.from_numpy(batch_np).to(device, non_blocking=True)
        preds    = classify_batch(batch_t, classifier, device=device)
        for i, pos in enumerate(preds):
            is_positive[start + i] = bool(pos)

    # ── Stage 2: segment positive tiles only ────────────────────────────
    pos_indices = [i for i, p in enumerate(is_positive) if p]
    if not pos_indices:
        return is_positive, masks

    for start in range(0, len(pos_indices), batch_size):
        chunk   = pos_indices[start : start + batch_size]
        batch_np = np.stack([tile_arrays[i] for i in chunk], axis=0)
        batch_t  = torch.from_numpy(batch_np).to(device, non_blocking=True)
        seg_masks = segment_batch(batch_t, segmenter)
        for local_i, global_i in enumerate(chunk):
            masks[global_i] = seg_masks[local_i]

    return is_positive, masks

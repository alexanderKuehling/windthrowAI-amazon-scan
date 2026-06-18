"""Model definitions and weight loading for the two-stage inference pipeline."""

from pathlib import Path
from typing import Optional

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn

from .config import (
    CLASSIFIER_BACKBONE,
    CLASSIFIER_DROPOUT,
    CLASSIFIER_PTH,
    NUM_MODEL_BANDS,
    SEGMENTER_BACKBONE,
    SEGMENTER_PTH,
)


# ── Stage-1 : lightweight classifier ────────────────────────────────────

class WindthrowClassifier(nn.Module):
    """
    MobileNetV3-Small backbone + single linear head.
    ~2.5 M params, ~0.11 ms @ 256 px  (torch.compiled, bs=64 on A100).
    """

    def __init__(
        self,
        backbone_name: str = CLASSIFIER_BACKBONE,
        in_channels:   int = NUM_MODEL_BANDS,
        dropout:       float = CLASSIFIER_DROPOUT,
        pretrained:    bool = False,   # False at inference: weights come from .pth
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained  = pretrained,
            in_chans    = in_channels,
            num_classes = 0,           # returns pooled feature vector
        )
        with torch.no_grad():
            dummy       = torch.zeros(1, in_channels, 64, 64)
            num_features = self.backbone(dummy).shape[-1]
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)   # (B,)


# ── Stage-2 : segmenter ──────────────────────────────────────────────────

class WindthrowSegmenter(nn.Module):
    """UNet with ResNet-34 encoder; trained on positive tiles only."""

    def __init__(
        self,
        encoder_name: str = SEGMENTER_BACKBONE,
        in_channels:  int = NUM_MODEL_BANDS,
    ):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name    = encoder_name,
            encoder_weights = None,    # no ImageNet init at inference
            in_channels     = in_channels,
            classes         = 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)            # (B, 1, H, W)  – raw logits


# ── Weight loading ───────────────────────────────────────────────────────

def _unwrap_state_dict(checkpoint: dict) -> dict:
    """
    Accept both formats:
      • new format: {"state_dict": {...}, "p5": ..., ...}  (training + gradio export)
      • legacy format: raw OrderedDict of weights
    Strips _orig_mod. (torch.compile) and module. (DataParallel) prefixes.
    """
    state = checkpoint.get("state_dict", checkpoint)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def _maybe_data_parallel(model: nn.Module, device: torch.device) -> nn.Module:
    """Wrap with DataParallel when >1 CUDA device is visible."""
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()}× GPUs via DataParallel")
        return nn.DataParallel(model)
    return model


def load_classifier(
    path:          Path         = CLASSIFIER_PTH,
    device:        torch.device = torch.device("cuda"),
    compile_model: bool         = True,
) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    backbone = checkpoint.get("backbone", CLASSIFIER_BACKBONE) if isinstance(checkpoint, dict) else CLASSIFIER_BACKBONE
    dropout  = checkpoint.get("dropout",  CLASSIFIER_DROPOUT)  if isinstance(checkpoint, dict) else CLASSIFIER_DROPOUT
    model    = WindthrowClassifier(backbone_name=backbone, dropout=dropout)
    model.load_state_dict(_unwrap_state_dict(checkpoint) if isinstance(checkpoint, dict) else checkpoint)
    model.to(device).eval()
    model = _maybe_data_parallel(model, device)
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    return model


def load_segmenter(
    path:          Path         = SEGMENTER_PTH,
    device:        torch.device = torch.device("cuda"),
    compile_model: bool         = False,
) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_state  = _unwrap_state_dict(checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Checkpoint was saved directly from smp.Unet (no WindthrowSegmenter wrapper),
    # so keys have no "unet." prefix – add it to match our module structure.
    if not any(k.startswith("unet.") for k in raw_state):
        raw_state = {"unet." + k: v for k, v in raw_state.items()}

    model = WindthrowSegmenter()
    model.load_state_dict(raw_state)
    model.to(device).eval()
    model = _maybe_data_parallel(model, device)
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    return model

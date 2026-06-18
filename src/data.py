"""Landsat-8 tile download via Google Earth Engine + on-the-fly NBR."""

import io
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple

import ee
import numpy as np
import rasterio
import rasterio.transform
import requests
import torch

from .config import (
    CLOUD_COVER_MAX,
    DATE_FROM,
    DATE_TO,
    GEE_KEY_PATH,
    GEE_PROJECT,
    NORM_P5,
    NORM_P95,
    RESOLUTION_M,
    TILE_SIZE_PX,
)

# GEE Landsat-8 C02 L2 bands → model input order: R, G, B, NIR, SWIR1, SWIR2
GEE_L8_BANDS   = ["SR_B4", "SR_B3", "SR_B2", "SR_B5", "SR_B6", "SR_B7"]
GEE_COLLECTION = "LANDSAT/LC08/C02/T1_L2"


# ── GEE initialisation ────────────────────────────────────────────────────

def init_gee(key_path: Optional[Path] = None, key_env: str = "GEE_KEY_FILE") -> None:
    """
    Initialise GEE with the service-account credentials.
    Resolution order: *key_path* arg → env-var *key_env* → GEE_KEY_PATH from config.
    """
    import os
    if key_path is None:
        key_path = GEE_KEY_PATH

    if key_path and key_path.exists():
        key_data = json.loads(key_path.read_text())
    else:
        raw = os.environ.get(key_env, "")
        if not raw:
            raise EnvironmentError(
                f"GEE key not found. Place it at {GEE_KEY_PATH} "
                f"or set the env-var {key_env}."
            )
        key_data = json.loads(raw)

    creds = ee.ServiceAccountCredentials(
        email    = key_data["client_email"],
        key_data = json.dumps(key_data),
    )
    ee.Initialize(creds, project=GEE_PROJECT)


# ── Normalization (mirrors preprocess() in windthrowAI-classification-gradio) ─

def preprocess(
    ms:  np.ndarray,
    p5:  np.ndarray = NORM_P5,
    p95: np.ndarray = NORM_P95,
) -> torch.Tensor:
    """
    Per-band percentile clip + scale to [0, 1], then return (1, C, H, W) tensor.

    ms : float32 (C, H, W)  raw Landsat-8 DN values
    Uses fixed training-time p5/p95 – never recomputed on inference data.
    """
    hwc = np.moveaxis(ms, 0, -1).copy()  # (H, W, C)
    hwc = np.nan_to_num(hwc.astype(np.float32), nan=0.0)
    for i in range(hwc.shape[2]):
        band     = np.clip(hwc[:, :, i], p5[i], p95[i])
        hwc[:, :, i] = (band - p5[i]) / (p95[i] - p5[i] + 1e-8)
    return torch.from_numpy(np.ascontiguousarray(hwc)).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)


# ── GEE composite ─────────────────────────────────────────────────────────

def _build_composite(
    bbox:      Tuple[float, float, float, float],
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
) -> Optional[ee.Image]:
    """
    Least-cloudy Landsat-8 mosaic for *bbox*.
    Sorted ascending by cloud cover so mosaic() picks the clearest valid pixel.
    Returns None if no scenes are available.
    """
    region = ee.Geometry.Rectangle(list(bbox))
    col = (
        ee.ImageCollection(GEE_COLLECTION)
        .filterBounds(region)
        .filterDate(date_from, date_to)
        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
        .select(GEE_L8_BANDS)
        .sort("CLOUD_COVER")
    )
    if col.size().getInfo() == 0:
        return None
    mosaic = col.mosaic().clip(region)
    return mosaic.multiply(0.0000275).add(-0.2)


def _add_nbr(image: ee.Image) -> ee.Image:
    nbr    = image.normalizedDifference(["SR_B5", "SR_B7"]).rename("NBR")
    nbr_01 = nbr.add(1.0).multiply(0.5)
    return image.addBands(nbr_01)


# ── Download one tile ─────────────────────────────────────────────────────

def download_tile(
    bbox:      Tuple[float, float, float, float],
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
    tile_size: int = TILE_SIZE_PX,
    scale:     int = RESOLUTION_M,
) -> Optional[np.ndarray]:
    """
    Download a Landsat-8 tile from GEE.

    Returns raw float32 (7, tile_size, tile_size) with bands
    [R, G, B, NIR, SWIR1, SWIR2, NBR], or None if no data is available.
    Call preprocess() on the result before feeding it to the models.
    """
    composite = _build_composite(bbox, date_from, date_to, max_cloud)
    if composite is None:
        return None

    img    = _add_nbr(composite)
    region = ee.Geometry.Rectangle(list(bbox))

    try:
        url  = img.getDownloadURL({
            "bands":  GEE_L8_BANDS + ["NBR"],
            "region": region,
            "scale":  scale,
            "format": "NPY",
        })
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()

        arr      = np.load(io.BytesIO(resp.content))  # structured array, keyed by band name
        channels = [arr[b].astype(np.float32) for b in (GEE_L8_BANDS + ["NBR"])]
        ms       = np.stack(channels, axis=0)          # (7, H, W)

        if ms.shape[1] != tile_size or ms.shape[2] != tile_size:
            import cv2
            ms = np.stack([
                cv2.resize(ms[i], (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                for i in range(ms.shape[0])
            ], axis=0)

        return np.nan_to_num(ms, nan=0.0)

    except Exception as exc:
        warnings.warn(f"GEE download failed for bbox {bbox}: {exc}")
        return None


# ── GeoTIFF output ───────────────────────────────────────────────────────

def save_mask(
    mask: np.ndarray,
    bbox: Tuple[float, float, float, float],
    path: Path,
) -> None:
    """Write a binary (H, W) uint8 mask as a compressed WGS-84 GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    transform = rasterio.transform.from_bounds(
        lon_min, lat_min, lon_max, lat_max, mask.shape[1], mask.shape[0]
    )
    with rasterio.open(
        path, "w",
        driver    = "GTiff",
        height    = mask.shape[0],
        width     = mask.shape[1],
        count     = 1,
        dtype     = rasterio.uint8,
        crs       = "EPSG:4326",
        transform = transform,
        compress  = "lzw",
    ) as dst:
        dst.write(mask.astype(np.uint8), 1)

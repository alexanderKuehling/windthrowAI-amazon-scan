"""Landsat-8 tile download via Google Earth Engine + on-the-fly NBR."""

import io
import json
import warnings
from datetime import datetime, timezone
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

def _collection_stats(col: ee.ImageCollection) -> dict:
    """Scene count, acquisition date range and mean scene cloud cover of *col*.

    Fetched in a single getInfo() — the same number of GEE round-trips the size
    check alone used to cost — so recording the provenance of each composite is
    free. Falls back to the plain size query if the aggregate call fails.
    """
    try:
        info = ee.Dictionary({
            "n_scenes":   col.size(),
            "t_first":    col.aggregate_min("system:time_start"),
            "t_last":     col.aggregate_max("system:time_start"),
            "cloud_mean": col.aggregate_mean("CLOUD_COVER"),
        }).getInfo()
    except Exception as exc:  # noqa: BLE001 — provenance is optional, the scan is not
        warnings.warn(f"GEE composite stats unavailable: {exc}")
        return {"n_scenes": int(col.size().getInfo()), "date_first": "",
                "date_last": "", "cloud_mean": ""}

    def _date(ms) -> str:
        if ms is None:
            return ""
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")

    cloud = info.get("cloud_mean")
    return {
        "n_scenes":   int(info.get("n_scenes") or 0),
        "date_first": _date(info.get("t_first")),
        "date_last":  _date(info.get("t_last")),
        "cloud_mean": round(float(cloud), 2) if cloud is not None else "",
    }


def _build_composite(
    bbox:      Tuple[float, float, float, float],
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
) -> Tuple[Optional[ee.Image], dict]:
    """
    Least-cloudy Landsat-8 mosaic for *bbox*, plus the composite's provenance
    (see _collection_stats). Sorted ascending by cloud cover so mosaic() picks
    the clearest valid pixel. The image is None if no scenes are available.
    """
    region = ee.Geometry.Rectangle(list(bbox))
    col = (
        ee.ImageCollection(GEE_COLLECTION)
        .filterBounds(region)
        .filterDate(date_from, date_to)
        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
        .map(_mask_landsat_clouds)
        .select(GEE_L8_BANDS)
        .sort("CLOUD_COVER", False)
    )
    stats = _collection_stats(col)
    if stats["n_scenes"] == 0:
        return None, stats
    mosaic = col.mosaic().clip(region)
    return mosaic.multiply(0.0000275).add(-0.2), stats


def _mask_landsat_clouds(image: "ee.Image") -> "ee.Image":
    qa = image.select("QA_PIXEL")
    dilated_cloud = 1 << 1
    cirrus = 1 << 2
    cloud = 1 << 3
    cloud_shadow = 1 << 4
    mask = (qa.bitwiseAnd(dilated_cloud).eq(0)
        .And(qa.bitwiseAnd(cirrus).eq(0))
        .And(qa.bitwiseAnd(cloud).eq(0))
        .And(qa.bitwiseAnd(cloud_shadow).eq(0)))
    return image.updateMask(mask)


def _add_nbr(image: ee.Image) -> ee.Image:
    nbr    = image.normalizedDifference(["SR_B5", "SR_B7"]).rename("NBR")
    nbr_01 = nbr.add(1.0).multiply(0.5)
    return image.addBands(nbr_01)


# ── Download one tile ─────────────────────────────────────────────────────

def valid_fraction(ms: np.ndarray) -> float:
    """Share of tile pixels that carry real observations, in [0, 1].

    Cloud/shadow pixels are removed by _mask_landsat_clouds and come back from GEE
    as NaN (or all-zero once filled), so a low value means the model saw mostly
    holes — the single most useful quality flag for triaging a detection.
    """
    if ms.size == 0:
        return 0.0
    valid = np.isfinite(ms).all(axis=0) & (ms != 0).any(axis=0)
    return round(float(valid.mean()), 4)


def download_tile_with_meta(
    bbox:      Tuple[float, float, float, float],
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
    tile_size: int = TILE_SIZE_PX,
    scale:     int = RESOLUTION_M,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Download a Landsat-8 tile from GEE together with its provenance.

    Returns (ms, meta): raw float32 (7, tile_size, tile_size) with bands
    [R, G, B, NIR, SWIR1, SWIR2, NBR] — or None if no data is available — and a
    dict with n_scenes / date_first / date_last / cloud_mean / valid_frac.
    Call preprocess() on the array before feeding it to the models.
    """
    composite, meta = _build_composite(bbox, date_from, date_to, max_cloud)
    meta["valid_frac"] = ""
    if composite is None:
        return None, meta

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

        meta["valid_frac"] = valid_fraction(ms)  # before nan_to_num: holes are still NaN
        return np.nan_to_num(ms, nan=0.0), meta

    except Exception as exc:
        warnings.warn(f"GEE download failed for bbox {bbox}: {exc}")
        return None, meta


def download_tile(
    bbox:      Tuple[float, float, float, float],
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
    tile_size: int = TILE_SIZE_PX,
    scale:     int = RESOLUTION_M,
) -> Optional[np.ndarray]:
    """Imagery-only form of download_tile_with_meta (see there)."""
    ms, _ = download_tile_with_meta(bbox, date_from, date_to, max_cloud, tile_size, scale)
    return ms


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


def save_prob_map(
    prob: np.ndarray,
    bbox: Tuple[float, float, float, float],
    path: Path,
) -> None:
    """Write the segmenter's per-pixel probability (H, W) as a uint8 GeoTIFF.

    Quantised to 1/255 with the scale stored on the band, so probability = DN/255.
    ~25 kB per tile against ~260 kB for float32, and it makes the segmentation
    threshold a post-processing choice: re-thresholding a saved probability map
    costs nothing, while a saved binary mask can only be redone on the GPU.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    transform = rasterio.transform.from_bounds(
        lon_min, lat_min, lon_max, lat_max, prob.shape[1], prob.shape[0]
    )
    dn = np.clip(np.rint(prob.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    with rasterio.open(
        path, "w",
        driver    = "GTiff",
        height    = prob.shape[0],
        width     = prob.shape[1],
        count     = 1,
        dtype     = rasterio.uint8,
        crs       = "EPSG:4326",
        transform = transform,
        compress  = "deflate",
    ) as dst:
        dst.write(dn, 1)
        dst.scales = [1.0 / 255.0]


def save_input(
    ms: np.ndarray,
    bbox: Tuple[float, float, float, float],
    path: Path,
) -> None:
    """Write the raw (C, H, W) float32 model input (pre-normalization, same
    band order as BAND_NAMES) as a compressed WGS-84 GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    transform = rasterio.transform.from_bounds(
        lon_min, lat_min, lon_max, lat_max, ms.shape[2], ms.shape[1]
    )
    with rasterio.open(
        path, "w",
        driver    = "GTiff",
        height    = ms.shape[1],
        width     = ms.shape[2],
        count     = ms.shape[0],
        dtype     = rasterio.float32,
        crs       = "EPSG:4326",
        transform = transform,
        compress  = "lzw",
    ) as dst:
        dst.write(ms.astype(np.float32))

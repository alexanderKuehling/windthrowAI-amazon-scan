"""Landsat-8 tile download via Google Earth Engine + on-the-fly NBR.

GEE cost per tile: one computePixels request. The scene metadata that decides
whether a tile has imagery at all (and records its provenance) is fetched for
GEE_STATS_CHUNK tiles per request, and download_tiles() keeps GEE_WORKERS pixel
requests in flight — see the request budget in config.py.
"""

import json
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import ee
import numpy as np
import rasterio
import rasterio.transform

from .config import (
    CLOUD_COVER_MAX,
    DATE_FROM,
    DATE_TO,
    GEE_HIGH_VOLUME,
    GEE_KEY_PATH,
    GEE_PROJECT,
    GEE_RETRIES,
    GEE_STATS_CHUNK,
    GEE_WORKERS,
    NBR_ENCODING,
    NORM_P5,
    NORM_P95,
    TILE_SIZE_PX,
)

# GEE Landsat-8 C02 L2 bands → model input order: R, G, B, NIR, SWIR1, SWIR2
GEE_L8_BANDS   = ["SR_B4", "SR_B3", "SR_B2", "SR_B5", "SR_B6", "SR_B7"]
GEE_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
GEE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"

Bbox = Tuple[float, float, float, float]


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
    ee.Initialize(
        creds,
        project = GEE_PROJECT,
        url     = GEE_HIGH_VOLUME_URL if GEE_HIGH_VOLUME else None,
    )


# ── Normalization (mirrors preprocess() in windthrowAI-classification-gradio) ─

def preprocess(
    ms:  np.ndarray,
    p5:  np.ndarray = NORM_P5,
    p95: np.ndarray = NORM_P95,
) -> "torch.Tensor":
    """
    Per-band percentile clip + scale to [0, 1], then return (1, C, H, W) tensor.

    ms : float32 (C, H, W)  raw Landsat-8 DN values
    Uses fixed training-time p5/p95 – never recomputed on inference data.
    """
    import torch  # only here: the download code (and data_pc.py) runs without torch
    hwc = np.moveaxis(ms, 0, -1).copy()  # (H, W, C)
    hwc = np.nan_to_num(hwc.astype(np.float32), nan=0.0)
    for i in range(hwc.shape[2]):
        band     = np.clip(hwc[:, :, i], p5[i], p95[i])
        hwc[:, :, i] = (band - p5[i]) / (p95[i] - p5[i] + 1e-8)
    return torch.from_numpy(np.ascontiguousarray(hwc)).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)


# ── GEE requests ──────────────────────────────────────────────────────────

def _retry(fn, what: str, attempts: int = GEE_RETRIES):
    """Call fn(), backing off exponentially (with jitter) between attempts.

    The ee client already retries HTTP 429 itself (5×, ~30 s) inside every call.
    This covers what that does not: a quota burst that outlasts it, "Computation
    timed out", a connection dropped at the cluster proxy. The jitter keeps
    workers that failed together from retrying in lockstep."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — re-raised after the last attempt
            if attempt == attempts:
                raise
            wait = min(120.0, 10.0 * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
            warnings.warn(f"GEE {what} failed ({exc}) — retry {attempt}/{attempts - 1} in {wait:.0f}s")
            time.sleep(wait)


# ── GEE composite ─────────────────────────────────────────────────────────

def _scenes(date_from: str, date_to: str, max_cloud: int) -> ee.ImageCollection:
    """Candidate Landsat-8 scenes, before any per-tile spatial filter."""
    return (
        ee.ImageCollection(GEE_COLLECTION)
        .filterDate(date_from, date_to)
        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
    )


_UNKNOWN_STATS = {"n_scenes": "", "date_first": "", "date_last": "", "cloud_mean": ""}


def _format_stats(info: dict) -> dict:
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


def composite_stats(
    bboxes:    Sequence[Bbox],
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
) -> List[Optional[dict]]:
    """Scene count, acquisition date range and mean scene cloud cover behind
    each tile's composite, one getInfo() per GEE_STATS_CHUNK tiles instead of
    one per tile. Scene metadata only — no pixels are computed.

    n_scenes == 0 means GEE holds no imagery for the tile (open ocean, or nothing
    under the cloud limit in the date window). An entry is None when its chunk
    still failed after retries: unknown, not empty.
    """
    scenes = _scenes(date_from, date_to, max_cloud)

    def _one(tile: ee.Feature) -> ee.Feature:
        col = scenes.filterBounds(tile.geometry())
        return ee.Feature(None, {
            "n_scenes":   col.size(),
            "t_first":    col.aggregate_min("system:time_start"),
            "t_last":     col.aggregate_max("system:time_start"),
            "cloud_mean": col.aggregate_mean("CLOUD_COVER"),
        })

    out: List[Optional[dict]] = []
    for start in range(0, len(bboxes), GEE_STATS_CHUNK):
        chunk = bboxes[start : start + GEE_STATS_CHUNK]
        tiles = ee.FeatureCollection(
            [ee.Feature(ee.Geometry.Rectangle(list(b))) for b in chunk]
        )
        try:
            feats = _retry(lambda: tiles.map(_one).getInfo()["features"],
                           f"scene stats ({len(chunk)} tiles)")
            if len(feats) != len(chunk):
                raise RuntimeError(f"got {len(feats)} results for {len(chunk)} tiles")
            out.extend(_format_stats(f.get("properties") or {}) for f in feats)
        except Exception as exc:  # noqa: BLE001 — these tiles are retried on resume
            warnings.warn(f"GEE scene stats failed for {len(chunk)} tiles: {exc}")
            out.extend([None] * len(chunk))
    return out


def _build_composite(
    bbox:      Bbox,
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
) -> ee.Image:
    """
    Least-cloudy Landsat-8 mosaic for *bbox*, scaled to reflectance. Sorted by
    cloud cover descending because mosaic() puts the last image on top, so the
    clearest valid pixel wins. Only meaningful for a tile with n_scenes > 0.
    No clip: the computePixels grid already bounds the output to the tile.
    """
    return (
        _scenes(date_from, date_to, max_cloud)
        .filterBounds(ee.Geometry.Rectangle(list(bbox)))
        .map(_mask_landsat_clouds)
        .select(GEE_L8_BANDS)
        .sort("CLOUD_COVER", False)
        .mosaic()
        .multiply(0.0000275).add(-0.2)
    )


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
    """NBR from NIR and SWIR2, in config.NBR_ENCODING (raw -1..1, as the models
    were trained, or the old (NBR + 1) / 2)."""
    nbr = image.normalizedDifference(["SR_B5", "SR_B7"]).rename("NBR")
    if NBR_ENCODING == "rescaled":
        nbr = nbr.add(1.0).multiply(0.5)
    return image.addBands(nbr)


# ── Download tiles ────────────────────────────────────────────────────────

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


def _fetch_pixels(
    bbox:      Bbox,
    date_from: str,
    date_to:   str,
    max_cloud: int,
    tile_size: int,
) -> np.ndarray:
    """One computePixels request → raw float32 (7, tile_size, tile_size).

    The output grid is the tile's bbox cut into tile_size² EPSG:4326 pixels —
    the same georeferencing save_input()/save_mask() write — so GEE returns the
    final shape directly and no resampling happens on our side."""
    lon_min, lat_min, lon_max, lat_max = bbox
    bands = GEE_L8_BANDS + ["NBR"]
    arr = ee.data.computePixels({
        "expression": _add_nbr(_build_composite(bbox, date_from, date_to, max_cloud)).select(bands),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": tile_size, "height": tile_size},
            "affineTransform": {
                "scaleX": (lon_max - lon_min) / tile_size, "shearX": 0, "translateX": lon_min,
                "shearY": 0, "scaleY": -(lat_max - lat_min) / tile_size, "translateY": lat_max,
            },
            "crsCode": "EPSG:4326",
        },
    })
    return np.stack([arr[b].astype(np.float32) for b in bands], axis=0)  # (7, H, W)


def download_tiles(
    bboxes:    Sequence[Bbox],
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
    tile_size: int = TILE_SIZE_PX,
    workers:   int = GEE_WORKERS,
) -> Iterator[Tuple[Optional[np.ndarray], dict]]:
    """
    Download Landsat-8 tiles from GEE, yielding (ms, meta) per bbox, in order.

    ms is raw float32 (7, tile_size, tile_size) with bands [R, G, B, NIR, SWIR1,
    SWIR2, NBR] — or None if there is no imagery or the download failed — and
    meta holds n_scenes / date_first / date_last / cloud_mean / valid_frac.
    Call preprocess() on the array before feeding it to the models.

    Costs one metadata request per GEE_STATS_CHUNK tiles, then one pixel request
    per tile that has imagery, *workers* of them in flight at once. A tile
    without imagery comes back with n_scenes == 0; a failed one with a non-zero
    or empty n_scenes, which is what results.is_retryable() tells apart.
    """
    stats = composite_stats(bboxes, date_from, date_to, max_cloud)

    def _one(i: int) -> Tuple[Optional[np.ndarray], dict]:
        bbox = bboxes[i]
        meta = dict(stats[i] or _UNKNOWN_STATS, valid_frac="")
        if stats[i] is None or meta["n_scenes"] == 0:
            return None, meta
        try:
            ms = _retry(
                lambda: _fetch_pixels(bbox, date_from, date_to, max_cloud, tile_size),
                f"download {bbox}",
            )
        except Exception as exc:  # noqa: BLE001 — the tile is retried on resume
            warnings.warn(f"GEE download failed for bbox {bbox}: {exc}")
            return None, meta
        meta["valid_frac"] = valid_fraction(ms)  # before nan_to_num: holes are still NaN
        return np.nan_to_num(ms, nan=0.0), meta

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        yield from pool.map(_one, range(len(bboxes)))


def download_tile_with_meta(
    bbox:      Bbox,
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
    tile_size: int = TILE_SIZE_PX,
) -> Tuple[Optional[np.ndarray], dict]:
    """Single-tile form of download_tiles (see there)."""
    return next(download_tiles([bbox], date_from, date_to, max_cloud, tile_size, workers=1))


def download_tile(
    bbox:      Bbox,
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
    tile_size: int = TILE_SIZE_PX,
) -> Optional[np.ndarray]:
    """Imagery-only form of download_tiles (see there)."""
    ms, _ = download_tile_with_meta(bbox, date_from, date_to, max_cloud, tile_size)
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

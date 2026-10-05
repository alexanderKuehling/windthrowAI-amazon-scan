"""Landsat-8 tiles from Microsoft Planetary Computer — the scan's GEE-free source.

Planetary Computer's `landsat-c2-l2` collection holds the same USGS Collection 2
Level-2 scenes as GEE's LANDSAT/LC08/C02/T1_L2 (scene ids
LC08_L2SP_<path><row>_<date>_02_T1), published as public Cloud-Optimized
GeoTIFFs. Instead of asking GEE to composite a tile, this module reads the
pixels straight from those files and composites them here, following
src/data.py step for step:

  scenes  Landsat 8, Tier 1, scene CLOUD_COVER < max_cloud, acquired in
          [date_from, date_to) (GEE's filterDate excludes the end date), whose
          footprint intersects the tile
  mask    a pixel is no data where QA_PIXEL flags dilated cloud, cirrus, cloud
          or cloud shadow (bits 1-4, as _mask_landsat_clouds), or fill (bit 0),
          or a band holds the fill value 0
  mosaic  least cloudy scene on top: per pixel, the first valid value in
          ascending CLOUD_COVER order, ties to the later scene id — GEE's
          .sort("CLOUD_COVER", False).mosaic() (see mosaic_order)
  grid    the tile's bbox cut into tile_size² EPSG:4326 pixels, nearest
          neighbour — the computePixels grid in data._fetch_pixels
  values  reflectance = DN * 0.0000275 - 0.2; NBR = (B5 - B7) / (B5 + B7), in
          config.NBR_ENCODING (raw, or the old (NBR + 1) / 2) — data._add_nbr

The source values are therefore identical; what can differ is which source
pixel nearest-neighbour picks right at a pixel boundary, and the order of two
equally cloudy scenes. scripts/compare_sources.py measures both against GEE.

Cost: no compute quota. One STAC search per download_tiles() call (the scan
calls it per chunk of DOWNLOAD_CHUNK tiles); per tile, scenes are read least
cloudy first and reading stops once every pixel has a value, so a clear tile
costs one scene's QA band and six reflectance bands, each a few range requests.
download_tiles() has data.download_tiles' signature and result contract, so
run_inference.py and merge_results.py treat both sources alike.
"""

import os
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    CLOUD_COVER_MAX,
    DATE_FROM,
    DATE_TO,
    NBR_ENCODING,
    PC_COLLECTION,
    PC_RETRIES,
    PC_STAC_URL,
    PC_WORKERS,
    TILE_SIZE_PX,
)
from .data import valid_fraction

# GDAL settings for reading COGs over HTTPS: don't list the remote "directory"
# on open, merge adjacent range requests, cache blocks in memory, and retry
# throttled/transient HTTP errors itself. Set before any file is opened; GDAL
# reads unset config options from the environment.
for _key, _value in {
    "GDAL_DISABLE_READDIR_ON_OPEN":       "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE":                          "TRUE",
    "GDAL_HTTP_MAX_RETRY":                "4",
    "GDAL_HTTP_RETRY_DELAY":              "2",
}.items():
    os.environ.setdefault(_key, _value)

# Asset keys in data.GEE_L8_BANDS order: SR_B4, SR_B3, SR_B2, SR_B5, SR_B6, SR_B7.
PC_BANDS = ["red", "green", "blue", "nir08", "swir16", "swir22"]
QA_ASSET = "qa_pixel"
QA_FILL  = 1 << 0
QA_CLOUD = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)   # data._mask_landsat_clouds
SR_SCALE, SR_OFFSET = 0.0000275, -0.2

Bbox = Tuple[float, float, float, float]


@dataclass
class Scene:
    id:     str
    cloud:  float
    day:    str               # "YYYY-MM-DD"
    geom:   object            # shapely footprint
    hrefs:  Dict[str, str]    # asset key -> unsigned href


def _retry(fn, what: str, attempts: int = PC_RETRIES):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — re-raised after the last attempt
            if attempt == attempts:
                raise
            wait = min(60.0, 5.0 * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
            warnings.warn(f"Planetary Computer {what} failed ({exc}) — retry {attempt}/{attempts - 1} in {wait:.0f}s")
            time.sleep(wait)


def _datetime_range(date_from: str, date_to: str) -> str:
    """STAC interval matching GEE's filterDate(date_from, date_to): end excluded."""
    end = datetime.combine(date.fromisoformat(date_to), datetime.min.time()) - timedelta(seconds=1)
    return f"{date_from}T00:00:00Z/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def search_scenes(bboxes: Sequence[Bbox], date_from: str, date_to: str, max_cloud: int) -> List[Scene]:
    """Every candidate scene over the union of *bboxes* (one paged STAC search)."""
    import pystac_client
    from shapely.geometry import shape

    union = [min(b[0] for b in bboxes), min(b[1] for b in bboxes),
             max(b[2] for b in bboxes), max(b[3] for b in bboxes)]

    def _search():
        client = pystac_client.Client.open(PC_STAC_URL)
        return list(client.search(
            collections = [PC_COLLECTION],
            bbox        = union,
            datetime    = _datetime_range(date_from, date_to),
            query       = {
                "platform":                   {"in": ["landsat-8"]},
                "landsat:collection_category": {"in": ["T1"]},
                "eo:cloud_cover":              {"lt": max_cloud},
            },
            limit = 500,
        ).items())

    items = _retry(_search, f"scene search over {union}")
    return [
        Scene(
            id    = it.id,
            cloud = float(it.properties["eo:cloud_cover"]),
            day   = it.datetime.astimezone(timezone.utc).strftime("%Y-%m-%d"),
            geom  = shape(it.geometry),
            hrefs = {k: it.assets[k].href for k in PC_BANDS + [QA_ASSET]},
        )
        for it in items
    ]


def _stats(hits: List[Scene]) -> dict:
    """Provenance in data._format_stats' shape."""
    return {
        "n_scenes":   len(hits),
        "date_first": min(s.day for s in hits) if hits else "",
        "date_last":  max(s.day for s in hits) if hits else "",
        "cloud_mean": round(sum(s.cloud for s in hits) / len(hits), 2) if hits else "",
    }


def _read(href: str, bbox: Bbox, size: int, nodata: int) -> np.ndarray:
    """One band of one scene resampled onto the tile grid (nearest neighbour).
    Pixels outside the scene come back as *nodata*."""
    import planetary_computer
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT

    transform = from_bounds(*bbox, size, size)
    with rasterio.open(planetary_computer.sign(href)) as src, WarpedVRT(
        src, crs="EPSG:4326", transform=transform, width=size, height=size,
        resampling=Resampling.nearest, nodata=nodata,
    ) as vrt:
        return vrt.read(1)


def mosaic_order(hits: List[Scene]) -> List[Scene]:
    """Scenes in the order GEE stacks them, top first: least cloudy, and among
    equally cloudy scenes (common — a dry-season Amazon window has hundreds at
    exactly 0%) the one later in the collection, i.e. the later scene id (path,
    row, then date). GEE's stable descending sort keeps ties in collection
    order, and mosaic() puts the last image on top."""
    return sorted(sorted(hits, key=lambda s: s.id, reverse=True), key=lambda s: s.cloud)


def composite_tile(bbox: Bbox, hits: List[Scene], tile_size: int) -> np.ndarray:
    """Raw float32 (7, tile_size, tile_size) [R, G, B, NIR, SWIR1, SWIR2, NBR],
    NaN where no scene had a valid pixel (see the module docstring)."""
    out    = np.full((len(PC_BANDS), tile_size, tile_size), np.nan, dtype=np.float32)
    filled = np.zeros((tile_size, tile_size), dtype=bool)
    for scene in mosaic_order(hits):
        qa    = _read(scene.hrefs[QA_ASSET], bbox, tile_size, nodata=QA_FILL)
        fresh = (qa != 0) & ((qa & QA_FILL) == 0) & ((qa & QA_CLOUD) == 0) & ~filled
        if not fresh.any():
            continue
        dn = np.stack([_read(scene.hrefs[b], bbox, tile_size, nodata=0) for b in PC_BANDS])
        fresh &= (dn != 0).all(axis=0)
        out[:, fresh] = dn[:, fresh].astype(np.float32) * SR_SCALE + SR_OFFSET
        filled |= fresh
        if filled.all():
            break

    nir, swir2 = out[3], out[5]
    total = nir + swir2
    with np.errstate(invalid="ignore", divide="ignore"):
        nbr = np.where(total != 0, (nir - swir2) / total, 0.0)
    if NBR_ENCODING == "rescaled":
        nbr = (nbr + 1.0) * 0.5
    nbr = np.where(np.isfinite(total), nbr, np.nan).astype(np.float32)
    return np.concatenate([out, nbr[None]], axis=0)


def download_tiles(
    bboxes:    Sequence[Bbox],
    date_from: str = DATE_FROM,
    date_to:   str = DATE_TO,
    max_cloud: int = CLOUD_COVER_MAX,
    tile_size: int = TILE_SIZE_PX,
    workers:   int = PC_WORKERS,
) -> Iterator[Tuple[Optional[np.ndarray], dict]]:
    """
    Planetary Computer counterpart of data.download_tiles, with the same
    contract: yields (ms, meta) per bbox, in order. ms is raw float32
    (7, tile_size, tile_size), NaN filled with 0, or None when the tile has no
    scene (n_scenes == 0, final) or its download failed (n_scenes non-zero, or
    empty when the scene search itself failed — results.is_retryable() retries
    both on resume).
    """
    from shapely.geometry import box

    try:
        scenes = search_scenes(bboxes, date_from, date_to, max_cloud)
    except Exception as exc:  # noqa: BLE001 — these tiles are retried on resume
        warnings.warn(f"Planetary Computer scene search failed for {len(bboxes)} tiles: {exc}")
        scenes = None

    def _one(bbox: Bbox) -> Tuple[Optional[np.ndarray], dict]:
        if scenes is None:
            return None, {"n_scenes": "", "date_first": "", "date_last": "", "cloud_mean": "", "valid_frac": ""}
        tile = box(*bbox)
        hits = [s for s in scenes if s.geom.intersects(tile)]
        meta = dict(_stats(hits), valid_frac="")
        if not hits:
            return None, meta
        try:
            ms = _retry(lambda: composite_tile(bbox, hits, tile_size), f"download {bbox}")
        except Exception as exc:  # noqa: BLE001 — the tile is retried on resume
            warnings.warn(f"Planetary Computer download failed for bbox {bbox}: {exc}")
            return None, meta
        meta["valid_frac"] = valid_fraction(ms)  # before nan_to_num: holes are still NaN
        return np.nan_to_num(ms, nan=0.0), meta

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        yield from pool.map(_one, bboxes)

"""Generate and serialise a regular WGS-84 tile grid covering the Amazon."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import AMAZON_BBOX, OVERLAP_PX, RESOLUTION_M, TILE_SIZE_PX


@dataclass
class Tile:
    tile_id:   str
    row:       int
    col:       int
    lon_min:   float
    lat_min:   float
    lon_max:   float
    lat_max:   float

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)


def _px_to_deg(px: int, resolution_m: int, lat_center: float) -> Tuple[float, float]:
    """Return (delta_lon_deg, delta_lat_deg) for *px* pixels at *resolution_m*."""
    meters = px * resolution_m
    delta_lat = meters / 111_320.0
    delta_lon = meters / (111_320.0 * math.cos(math.radians(lat_center)))
    return delta_lon, delta_lat


def generate_grid(
    bbox: Tuple[float, float, float, float] = AMAZON_BBOX,
    tile_size_px: int = TILE_SIZE_PX,
    resolution_m: int = RESOLUTION_M,
    overlap_px: int = OVERLAP_PX,
) -> List[Tile]:
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_center = (lat_min + lat_max) / 2.0

    step_px   = tile_size_px - overlap_px
    dlon_tile, dlat_tile = _px_to_deg(tile_size_px, resolution_m, lat_center)
    dlon_step, dlat_step = _px_to_deg(step_px,      resolution_m, lat_center)

    tiles: List[Tile] = []
    row = 0
    lat = lat_max
    while lat > lat_min:
        col = 0
        lon = lon_min
        while lon < lon_max:
            t_lon_max = min(lon + dlon_tile, lon_max)
            t_lat_min = max(lat - dlat_tile, lat_min)
            tiles.append(Tile(
                tile_id = f"r{row:04d}_c{col:04d}",
                row     = row,
                col     = col,
                lon_min = round(lon,         6),
                lat_min = round(t_lat_min,   6),
                lon_max = round(t_lon_max,   6),
                lat_max = round(lat,         6),
            ))
            lon += dlon_step
            col += 1
        lat -= dlat_step
        row += 1

    return tiles


def save_grid(tiles: List[Tile], path: Path) -> None:
    df = pd.DataFrame([{
        "tile_id": t.tile_id,
        "row":     t.row,
        "col":     t.col,
        "lon_min": t.lon_min,
        "lat_min": t.lat_min,
        "lon_max": t.lon_max,
        "lat_max": t.lat_max,
    } for t in tiles])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} tiles → {path}")


def load_grid(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_job_tiles(grid_path: Path, job_idx: int, tiles_per_job: int) -> pd.DataFrame:
    df   = load_grid(grid_path)
    start = job_idx * tiles_per_job
    return df.iloc[start : start + tiles_per_job].reset_index(drop=True)


def n_jobs(grid_path: Path, tiles_per_job: int) -> int:
    return math.ceil(len(load_grid(grid_path)) / tiles_per_job)

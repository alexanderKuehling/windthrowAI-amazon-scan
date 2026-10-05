"""Where the scan's Landsat-8 tiles come from, and how a task walks through them.

Two interchangeable sources with one contract, download_tiles(bboxes, ...) ->
(ms, meta) per tile (see src/data.py):
  "gee"  Google Earth Engine composites each tile (src/data.py)
  "pc"   Microsoft Planetary Computer's public files, composited here (src/data_pc.py)
config.DATA_SOURCE picks the default (env WINDTHROW_DATA_SOURCE overrides it).

download_in_chunks() is the loop run_inference.py and resegment.py share: it
downloads a task's tiles DOWNLOAD_CHUNK at a time and hands each chunk back
before fetching the next, so the caller classifies and writes it first. A task
killed at its walltime then loses one chunk, not every download it made.
"""

from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from .config import (
    DATA_SOURCE,
    DOWNLOAD_CHUNK,
    GEE_ARRAY_PARALLEL,
    GEE_WORKERS,
    PC_ARRAY_PARALLEL,
    PC_WORKERS,
)

SOURCES = ("gee", "pc")

Todo = Sequence[Tuple[str, Tuple[float, float, float, float]]]   # (tile_id, bbox)


def _check(source: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"unknown imagery source {source!r}; expected one of {SOURCES}")
    return source


def default_workers(source: str = DATA_SOURCE) -> int:
    """Tiles in flight per task: GEE's share of its 40-request budget, or PC's bandwidth share."""
    return GEE_WORKERS if _check(source) == "gee" else PC_WORKERS


def array_parallel(source: str = DATA_SOURCE) -> int:
    """Array tasks to run at once (the %N slurm/run_full_scan.sh submits with)."""
    return GEE_ARRAY_PARALLEL if _check(source) == "gee" else PC_ARRAY_PARALLEL


def get_downloader(source: str = DATA_SOURCE, key_path: Optional[Path] = None) -> Callable:
    """The source's download_tiles. "gee" initialises Earth Engine first."""
    if _check(source) == "pc":
        from .data_pc import download_tiles
        return download_tiles
    from .data import download_tiles, init_gee
    init_gee(key_path=key_path)
    return download_tiles


def download_in_chunks(
    todo:     Todo,
    download: Callable,
    workers:  int,
    chunk:    int = DOWNLOAD_CHUNK,
    **dl_kwargs,
) -> Iterator[List[Tuple[str, tuple, object, dict]]]:
    """Yield [(tile_id, bbox, ms, meta), ...] for consecutive chunks of *todo*."""
    for start in range(0, len(todo), chunk):
        part = todo[start : start + chunk]
        results = download([bbox for _, bbox in part], workers=workers, **dl_kwargs)
        yield [(tile_id, bbox, ms, meta) for (tile_id, bbox), (ms, meta) in zip(part, results)]

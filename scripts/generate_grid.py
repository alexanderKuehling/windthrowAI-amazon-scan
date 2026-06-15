"""
Generate the tile grid covering the Amazon and write it to tile_grid.parquet.

Run once (CPU-only, no GPU needed):
    python scripts/generate_grid.py

Output: tile_grid.parquet  (~55-100k rows, one per spatial tile)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import GRID_FILE, TILES_PER_JOB
from src.tile_grid import generate_grid, n_jobs, save_grid


def main():
    print("Generating tile grid for the Amazon …")
    tiles = generate_grid()
    save_grid(tiles, GRID_FILE)

    n = n_jobs(GRID_FILE, TILES_PER_JOB)
    print(f"Total tiles   : {len(tiles):,}")
    print(f"Tiles per job : {TILES_PER_JOB}")
    print(f"SLURM array   : 0–{n-1}  (--array=0-{n-1})")


if __name__ == "__main__":
    main()

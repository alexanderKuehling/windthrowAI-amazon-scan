"""
Visualize N positive tiles with their predicted masks overlaid in red.

    python scripts/visualize_positives.py <job_idx> [n]

Re-downloads the tile imagery via GEE (using the bbox stored in results.csv)
and overlays the saved segmentation mask. Saves PNGs to outputs/viz/.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import BAND_NAMES, NORM_P5, NORM_P95, OUTPUTS_DIR
from src.data import download_tile, init_gee


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <job_idx> [n=5]")
        sys.exit(1)

    job_idx = int(sys.argv[1])
    n       = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    job_dir  = OUTPUTS_DIR / f"job_{job_idx:04d}"
    csv_path = job_dir / "results.csv"

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["is_positive"] == "True" and row["mask_path"]:
                rows.append(row)

    if not rows:
        print("No positive tiles found.")
        return

    def mask_frac(row):
        return rasterio.open(ROOT / row["mask_path"]).read(1).mean()

    rows.sort(key=mask_frac, reverse=True)
    rows = rows[:n]

    init_gee()
    out_dir = OUTPUTS_DIR / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        tile_id = row["tile_id"]
        bbox    = tuple(float(row[k]) for k in ("lon_min", "lat_min", "lon_max", "lat_max"))

        ms = download_tile(bbox)
        if ms is None:
            print(f"  {tile_id}: download failed, skipping")
            continue

        rgb = np.stack([ms[0], ms[1], ms[2]], axis=-1)   # R, G, B reflectance (~0-0.3)
        rgb = np.clip(rgb / 0.3, 0, 1)                   # simple stretch for visibility

        mask    = rasterio.open(ROOT / row["mask_path"]).read(1)
        overlay = rgb.copy()
        overlay[mask == 1] = [1.0, 0.0, 0.0]              # mark positive pixels in red

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()

        for i, name in enumerate(BAND_NAMES):   # R, G, B, nIR, swIR1, swIR2, NBR
            band = np.clip(ms[i], NORM_P5[i], NORM_P95[i])
            band = (band - NORM_P5[i]) / (NORM_P95[i] - NORM_P5[i] + 1e-8)
            axes[i].imshow(band, cmap="gray", vmin=0, vmax=1)
            axes[i].set_title(name)
            axes[i].axis("off")

        axes[7].imshow(rgb)
        axes[7].set_title("RGB")
        axes[7].axis("off")

        axes[8].imshow(overlay)
        axes[8].set_title(f"mask overlay ({mask.mean()*100:.2f}%)")
        axes[8].axis("off")

        fig.suptitle(tile_id)
        plt.tight_layout()

        out_path = out_dir / f"{tile_id}_viz.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved {out_path}  ({mask.mean()*100:.2f}% positive pixels)")


if __name__ == "__main__":
    main()

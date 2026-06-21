"""
Visualize N resegmented tiles, comparing old vs new segmenter mask.

    python scripts/visualize_resegment.py <job_idx> [n]
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

from src.config import OUTPUTS_DIR
from src.data import download_tile, init_gee

RESEG_DIR = ROOT / "outputs_resegment"


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <job_idx> [n=5]")
        sys.exit(1)

    job_idx = int(sys.argv[1])
    n       = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    csv_path = RESEG_DIR / f"job_{job_idx:04d}" / "results.csv"

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["is_positive"] == "True" and row["mask_path"]:
                rows.append(row)

    if not rows:
        print("No positive tiles found in this resegment job.")
        return

    def mask_frac(row):
        return rasterio.open(ROOT / row["mask_path"]).read(1).mean()

    rows.sort(key=mask_frac, reverse=True)
    rows = rows[:n]

    init_gee()
    out_dir = RESEG_DIR / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        tile_id = row["tile_id"]
        bbox    = tuple(float(row[k]) for k in ("lon_min", "lat_min", "lon_max", "lat_max"))

        ms = download_tile(bbox)
        if ms is None:
            print(f"  {tile_id}: download failed, skipping")
            continue

        rgb = np.stack([ms[0], ms[1], ms[2]], axis=-1)
        rgb = np.clip(rgb / 0.3, 0, 1)

        new_mask = rasterio.open(ROOT / row["mask_path"]).read(1)
        new_overlay = rgb.copy()
        new_overlay[new_mask == 1] = [1.0, 0.0, 0.0]

        old_mask = None
        for old_path in OUTPUTS_DIR.glob(f"job_*/{tile_id}_mask.tif"):
            old_mask = rasterio.open(old_path).read(1)
            break

        n_panels = 3 if old_mask is not None else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[0].axis("off")

        panel = 1
        if old_mask is not None:
            old_overlay = rgb.copy()
            old_overlay[old_mask == 1] = [1.0, 0.0, 0.0]
            axes[panel].imshow(old_overlay)
            axes[panel].set_title(f"old mask ({old_mask.mean()*100:.2f}%)")
            axes[panel].axis("off")
            panel += 1

        axes[panel].imshow(new_overlay)
        axes[panel].set_title(f"new mask ({new_mask.mean()*100:.2f}%)")
        axes[panel].axis("off")

        fig.suptitle(tile_id)
        plt.tight_layout()

        out_path = out_dir / f"{tile_id}_viz.png"
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved {out_path}  (new={new_mask.mean()*100:.2f}%)")


if __name__ == "__main__":
    main()

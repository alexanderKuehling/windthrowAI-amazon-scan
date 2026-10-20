"""
Check that Microsoft Planetary Computer tiles (DATA_SOURCE "pc", the default,
src/data_pc.py) match the GEE tiles (DATA_SOURCE "gee").

    python scripts/compare_sources.py --live 40              # fresh tiles from both sources
    python scripts/compare_sources.py --from-outputs 200     # vs inputs a GEE scan already saved
    python scripts/compare_sources.py --from-outputs 200 --models   # + model decisions on both

--live downloads N tiles from a central Amazon box through both sources: N GEE
requests, so it runs on the login node in a minute but spends a little quota.
--from-outputs reads the GEE side from outputs/job_*/<tile>_input.tif (the
positives and spot checks a finished GEE scan saved) and makes no GEE request
at all. --models also runs the pinned classifier and segmenter
(model_versions.json, as step 01 does) on both versions of every tile.

Per tile it reports which share of pixels is identical in all 7 bands, the
median absolute difference per band where both have data, and both valid
fractions; with --models, whether the classifier decision agrees and the Dice
between the two masks. --live also counts the scenes GEE has over the sample
that Planetary Computer lacks. The table goes to outputs/compare_sources.csv,
the summary ends in a verdict:

  MATCH     >= 90% of tiles have >= 95% identical pixels, and (with --models)
            >= 98% of classifier decisions agree — the two are interchangeable
  DIFFERS   anything else — read the per-tile table, and scan with
            WINDTHROW_DATA_SOURCE=gee until it is understood

RESULT (2026-10-05, --live 20, without --models; its CSV is not in the repo):
  - 19 of 20 tiles are 99.4-99.98% bit-identical — the files are the same
    USGS product, and the rest are nearest-neighbour picks at pixel edges.
  - Scenes: for Jun-Sep 2024 over the Amazon box PC lacks 101 of GEE's 2,059
    Landsat-8 scenes (~5%, mostly single dates).
  - Verdict: MATCH by the rule above (19/20 = 95% of tiles >= 95% identical).
    Classifier agreement (--models) has not been measured yet.

A tile differs as a whole only when its best GEE scene is missing from
Planetary Computer, which then picks the next-best one.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    BAND_NAMES,
    DATE_FROM,
    DATE_TO,
    GRID_FILE,
    MODEL_VERSIONS_FILE,
    NBR_ENCODING,
    OUTPUTS_DIR,
    PC_WORKERS,
)
from src.sources import get_downloader

# Mostly land with imagery: the live sample is drawn from here, not the ocean
# and Andes corners of AMAZON_BBOX.
LIVE_REGION = (-75.0, -12.0, -50.0, 2.0)
MATCH_PIXEL_SHARE = 0.95     # a tile counts as identical from this share of pixels ...
MATCH_TILE_SHARE = 0.90      # ... and this share of tiles must be identical
MATCH_DECISION_SHARE = 0.98


def _hash(tile_id: str) -> str:
    return hashlib.md5(f"compare:{tile_id}".encode()).hexdigest()


def live_tiles(n: int):
    """n grid tiles inside LIVE_REGION, a fixed hash sample."""
    if GRID_FILE.exists():
        from src.tile_grid import load_grid
        grid = load_grid(GRID_FILE)
        rows = [(r.tile_id, (r.lon_min, r.lat_min, r.lon_max, r.lat_max)) for r in grid.itertuples()]
    else:
        from src.tile_grid import generate_grid
        rows = [(t.tile_id, t.bbox) for t in generate_grid()]
    lo_x, lo_y, hi_x, hi_y = LIVE_REGION
    inside = [r for r in rows if lo_x <= r[1][0] and r[1][2] <= hi_x and lo_y <= r[1][1] and r[1][3] <= hi_y]
    return sorted(inside, key=lambda r: _hash(r[0]))[:n]


def saved_tiles(n: int):
    """n tiles a GEE scan saved an input for: (tile_id, bbox, path, date_from, date_to, nbr_encoding)."""
    out = []
    for job_dir in sorted(OUTPUTS_DIR.glob("job_*")):
        meta_path = job_dir / "run_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("data_source", "gee") != "gee" or not (job_dir / "results.csv").exists():
            continue
        with open(job_dir / "results.csv", newline="") as f:
            for r in csv.DictReader(f):
                path = ROOT / r["input_path"] if r.get("input_path") else None
                if path and path.exists():
                    bbox = tuple(float(r[c]) for c in ("lon_min", "lat_min", "lon_max", "lat_max"))
                    out.append((r["tile_id"], bbox, path, meta.get("date_from", DATE_FROM),
                                meta.get("date_to", DATE_TO), meta.get("nbr_encoding") or "rescaled"))
    return sorted(out, key=lambda r: _hash(r[0]))[:n]


def catalog_gap(bboxes) -> tuple:
    """(n GEE scenes over the bboxes' union, ids of those Planetary Computer lacks)."""
    import ee
    from src.config import CLOUD_COVER_MAX
    from src.data import _scenes
    from src.data_pc import search_scenes
    union = [min(b[0] for b in bboxes), min(b[1] for b in bboxes), max(b[2] for b in bboxes), max(b[3] for b in bboxes)]
    gee = set(_scenes(DATE_FROM, DATE_TO, CLOUD_COVER_MAX).filterBounds(ee.Geometry.Rectangle(union))
              .aggregate_array("system:index").getInfo())
    # GEE ids are LC08_<pathrow>_<date>; PC's are LC08_L2SP_<pathrow>_<date>_02_T1
    pc = {s.id.replace("_L2SP", "")[:20] for s in search_scenes([union], DATE_FROM, DATE_TO, CLOUD_COVER_MAX)}
    return len(gee), sorted(gee - pc)


def _read_input(path: Path, nbr_encoding: str) -> np.ndarray:
    """A saved input, its NBR band converted to config.NBR_ENCODING (a scan from
    before the switch saved (NBR + 1) / 2)."""
    import rasterio
    with rasterio.open(path) as src:
        ms = src.read().astype(np.float32)
    if nbr_encoding != NBR_ENCODING:
        valid = (ms[:6] != 0).any(axis=0)
        ms[6] = np.where(valid, ms[6] * 2 - 1 if NBR_ENCODING == "raw" else (ms[6] + 1) / 2, 0)
    return ms


def compare_pixels(gee: np.ndarray, pc: np.ndarray) -> dict:
    valid_g = (gee != 0).any(axis=0)
    valid_p = (pc != 0).any(axis=0)
    both = valid_g & valid_p
    same = both & (np.abs(gee - pc) <= 1e-6).all(axis=0)
    row = {
        "valid_gee":       round(float(valid_g.mean()), 4),
        "valid_pc":        round(float(valid_p.mean()), 4),
        "validity_agrees": round(float((valid_g == valid_p).mean()), 4),
        "identical_share": round(float(same.sum() / both.sum()), 4) if both.any() else "",
    }
    for b, name in enumerate(BAND_NAMES):
        row[f"mad_{name}"] = round(float(np.median(np.abs(gee[b][both] - pc[b][both]))), 6) if both.any() else ""
    return row


class Models:
    """The pinned classifier + segmenter, as run_inference.py loads them."""

    def __init__(self):
        import torch
        from src.hub import download_checkpoint
        from src.models import load_classifier, load_segmenter
        versions = json.loads(MODEL_VERSIONS_FILE.read_text())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf_path = download_checkpoint(versions["classifier"]["repo_id"], versions["classifier"]["filename"],
                                       versions["classifier"]["revision"])
        seg_path = download_checkpoint(versions["segmenter"]["repo_id"], versions["segmenter"]["filename"],
                                       versions["segmenter"]["revision"])
        self.classifier = load_classifier(clf_path, self.device)
        self.segmenter, _ = load_segmenter(seg_path, self.device)

    def run(self, ms: np.ndarray):
        from src.config import CLASSIFIER_THRESHOLD, SEGMENTER_THRESHOLD
        from src.data import preprocess
        from src.pipeline import classify_probs, segment_probs
        x = preprocess(ms).to(self.device)
        prob = float(classify_probs(x, self.classifier)[0])
        mask = segment_probs(x, self.segmenter)[0] >= SEGMENTER_THRESHOLD
        return prob, prob >= CLASSIFIER_THRESHOLD, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", type=int, metavar="N", help="download N tiles from both GEE and PC")
    mode.add_argument("--from-outputs", type=int, metavar="N", help="compare N saved GEE inputs against PC")
    ap.add_argument("--models", action="store_true", help="also compare the pinned models' outputs")
    ap.add_argument("--workers", type=int, default=PC_WORKERS, help="concurrent PC tile downloads")
    ap.add_argument("--key-path", default=None, help="GEE service-account JSON (--live only)")
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "compare_sources.csv")
    args = ap.parse_args()

    if args.live:
        picked = [(t, b, None, DATE_FROM, DATE_TO, NBR_ENCODING) for t, b in live_tiles(args.live)]
        gee_dl = get_downloader("gee", key_path=Path(args.key_path) if args.key_path else None)
        gee_arrays = [ms for ms, _ in gee_dl([b for _, b, *_ in picked], workers=4)]
    else:
        picked = saved_tiles(args.from_outputs)
        if not picked:
            sys.exit(f"no saved GEE inputs under {OUTPUTS_DIR}/job_*/ — run a GEE scan first, or use --live")
        gee_arrays = [_read_input(p, enc) for _, _, p, _, _, enc in picked]

    pc_dl = get_downloader("pc")
    pc_arrays = []
    for date_from, date_to in sorted({(r[3], r[4]) for r in picked}):
        idx = [i for i, r in enumerate(picked) if (r[3], r[4]) == (date_from, date_to)]
        got = pc_dl([picked[i][1] for i in idx], date_from=date_from, date_to=date_to, workers=args.workers)
        pc_arrays.extend(zip(idx, (ms for ms, _ in got)))
    pc_by_idx = dict(pc_arrays)

    models = Models() if args.models else None
    rows = []
    for i, (tile_id, bbox, *_rest) in enumerate(picked):
        gee, pc = gee_arrays[i], pc_by_idx.get(i)
        row = {"tile_id": tile_id, "gee_ok": gee is not None, "pc_ok": pc is not None}
        if gee is not None and pc is not None:
            row.update(compare_pixels(gee, pc))
            if models:
                p_g, d_g, m_g = models.run(gee)
                p_p, d_p, m_p = models.run(pc)
                inter, total = float((m_g & m_p).sum()), float(m_g.sum() + m_p.sum())
                row.update({"clf_prob_gee": round(p_g, 4), "clf_prob_pc": round(p_p, 4),
                            "decision_agrees": d_g == d_p,
                            "mask_dice": round(2 * inter / total, 4) if total else 1.0})
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "tile_id", k))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    both = [r for r in rows if r.get("identical_share") not in (None, "")]
    print(f"{len(rows)} tiles: {sum(r['gee_ok'] for r in rows)} from GEE, {sum(r['pc_ok'] for r in rows)} from PC, "
          f"{len(both)} with data in both -> {args.out}")
    if not both:
        sys.exit("no tile had data from both sources — nothing to compare")
    share = float(np.median([r["identical_share"] for r in both]))
    tiles_same = float(np.mean([r["identical_share"] >= MATCH_PIXEL_SHARE for r in both]))
    print(f"identical pixels (median tile): {share:.1%}   tiles >= {MATCH_PIXEL_SHARE:.0%} identical: "
          f"{tiles_same:.0%}   validity agrees: {np.median([r['validity_agrees'] for r in both]):.1%}")
    print("median |GEE - PC| where both have data: " +
          ", ".join(f"{n} {np.median([r[f'mad_{n}'] for r in both]):.2g}" for n in BAND_NAMES))
    if args.live:
        n_gee, lacking = catalog_gap([b for _, b, *_ in picked])
        print(f"scenes over the sample: GEE {n_gee}, of which Planetary Computer lacks {len(lacking)}"
              + (f" (e.g. {', '.join(lacking[:3])})" if lacking else ""))
    ok = tiles_same >= MATCH_TILE_SHARE
    if models:
        agree = float(np.mean([r["decision_agrees"] for r in both]))
        dice = [r["mask_dice"] for r in both if r["clf_prob_gee"] >= 0.5 or r["clf_prob_pc"] >= 0.5]
        print(f"classifier decisions agree: {agree:.1%}" +
              (f"   mask Dice (flagged tiles, median): {np.median(dice):.3f}" if dice else ""))
        ok = ok and agree >= MATCH_DECISION_SHARE
    print("verdict: " + ("MATCH — Planetary Computer and GEE are interchangeable" if ok else
                         "DIFFERS — read the per-tile table; WINDTHROW_DATA_SOURCE=gee scans with GEE"))


if __name__ == "__main__":
    main()

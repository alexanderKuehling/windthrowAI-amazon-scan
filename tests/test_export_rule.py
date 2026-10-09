"""The export rule (src/results.py::is_detection, config.EXPORT_MIN_*): which
flagged tiles are published, and that the rest feeds the spot-check pool."""
import csv
import json

import numpy as np
import pytest

from src import results
from src.config import CLASSIFIER_THRESHOLD, EXPORT_MIN_PATCH_PX
from src.data import save_input, save_mask
from src.results import CSV_COLUMNS, base_row, is_control_sample, is_detection, largest_patch_px, patch_px_of

BBOX = (-60.0, -3.0, -59.93, -2.93)


def test_largest_patch_counts_one_8_connected_component():
    m = np.zeros((20, 20), dtype=np.uint8)
    m[0:3, 0:3] = 1            # 9 px
    m[3, 3] = 1                # touches diagonally -> same patch: 10
    m[10:14, 10:15] = 1        # 20 px, separate
    assert largest_patch_px(m) == 20
    m[14, 15] = 1              # diagonal neighbour of the 20-px block
    assert largest_patch_px(m) == 21
    assert largest_patch_px(np.zeros((5, 5))) == 0


def test_rule_needs_both_the_classifier_and_a_windthrow_sized_patch():
    assert is_detection(0.9, EXPORT_MIN_PATCH_PX)
    assert not is_detection(0.9, EXPORT_MIN_PATCH_PX - 1)
    assert not is_detection(CLASSIFIER_THRESHOLD - 0.01, 10_000)
    assert is_detection("", EXPORT_MIN_PATCH_PX)          # run from before clf_prob was recorded


def test_patch_falls_back_to_the_mask_file(tmp_path):
    m = np.zeros((16, 16), dtype=np.uint8)
    m[2:9, 2:9] = 1                                       # 49 px
    save_mask(m, BBOX, tmp_path / "m.tif")
    assert patch_px_of({"max_patch_px": "", "mask_path": "m.tif"}, tmp_path) == 49
    assert patch_px_of({"max_patch_px": "7", "mask_path": "m.tif"}, tmp_path) == 7   # recorded value wins
    assert patch_px_of({"max_patch_px": "", "mask_path": ""}, tmp_path) == 0


def _first_control_id(prefix):
    return next(f"{prefix}{i}" for i in range(100_000) if is_control_sample(f"{prefix}{i}", 1.0))


def test_export_publishes_detections_and_samples_dropped_tiles_as_spot_checks(tmp_path, monkeypatch):
    ex = pytest.importorskip("scripts.export_for_frontend")
    job = tmp_path / "outputs" / "job_0000"
    job.mkdir(parents=True)
    ms = np.random.rand(7, 16, 16).astype("float32")
    big, speck = np.zeros((16, 16), np.uint8), np.zeros((16, 16), np.uint8)
    big[:8, :8] = 1                                       # 64 px >= 30
    speck[0, 0] = 1                                       # 1 px
    sampled = _first_control_id("drop")                   # a dropped tile the spot-check hash picks
    rows = []
    for tile_id, mask in (("det", big), ("speck", speck), (sampled, speck)):
        save_mask(mask, BBOX, job / f"{tile_id}_mask.tif")
        save_input(ms, BBOX, job / f"{tile_id}_input.tif")
        r = base_row(tile_id, BBOX, {"n_scenes": 2, "valid_frac": 1.0})
        r.update(is_positive=True, clf_prob=0.9, mask_path=str(job / f"{tile_id}_mask.tif"),
                 input_path=str(job / f"{tile_id}_input.tif"))
        rows.append(results.fill_segmentation(r, mask.astype(float), mask))
    with open(job / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (job / "run_meta.json").write_text(json.dumps({"nbr_encoding": "raw"}))
    monkeypatch.setattr(ex, "ORIG_DIR", tmp_path / "outputs")
    monkeypatch.setattr(ex, "RESEG_DIR", tmp_path / "none")

    detections, dropped = ex.apply_export_rule(ex.collect_final_results())
    assert set(detections) == {"det"} and set(dropped) == {"speck", sampled}
    controls = ex.collect_control_tiles(detections, dropped)
    assert set(controls) == {sampled}
    c = controls[sampled]
    assert c["kind"] == "control" and c["mask_path"] is None and c["metrics"]["max_patch_px"] == "0"


def test_a_round_too_big_for_one_hf_folder_stops_before_any_upload(monkeypatch, capsys):
    ex = pytest.importorskip("scripts.export_for_frontend")
    monkeypatch.setattr(ex, "HF_MAX_FILES_PER_DIR", 3)
    flagged = {f"t{i}": {"metrics": {"clf_prob": "0.9", "max_patch_px": str(px)}}
               for i, px in enumerate((30, 40, 60, 120))}
    with pytest.raises(SystemExit):
        ex.check_folder_limit(flagged, n_controls=1)            # 4 detections + 1 spot check > 3
    out = capsys.readouterr().out
    assert "patch >=  50 px" in out and "patch >= 100 px (" in out
    flagged.pop("t0"), flagged.pop("t1")
    ex.check_folder_limit(flagged, n_controls=1)                # 2 + 1 fits: no exit

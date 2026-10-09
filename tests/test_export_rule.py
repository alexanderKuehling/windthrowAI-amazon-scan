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


def test_tile_files_go_into_one_subfolder_per_grid_row():
    ex = pytest.importorskip("scripts.export_for_frontend")
    assert ex.tile_file("inputs", "r0341_c0444") == "inputs/r0341/r0341_c0444.tif"
    assert ex.tile_file("masks", "r0002_c0010") == "masks/r0002/r0002_c0010.tif"
    assert len(ex.shard("odd-id")) == 2                          # hash fallback


def test_layout_check_stops_only_when_a_subfolder_would_overflow(monkeypatch):
    ex = pytest.importorskip("scripts.export_for_frontend")
    monkeypatch.setattr(ex, "HF_MAX_FILES_PER_DIR", 3)
    ex.check_folder_layout([f"r0001_c{c:04d}" for c in range(3)] + ["r0002_c0000"])   # 3 + 1: fits
    with pytest.raises(SystemExit):
        ex.check_folder_layout([f"r0001_c{c:04d}" for c in range(4)])                 # 4 in one row


def test_stale_remote_files_are_deleted_in_their_own_commit_before_the_upload(tmp_path, monkeypatch):
    ex = pytest.importorskip("scripts.export_for_frontend")
    staged = tmp_path / "inputs"
    staged.mkdir()
    for name in ("new.tif", "kept.tif"):
        (staged / name).write_bytes(b"x")
    calls = []

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            return ["inputs/old.tif", "inputs/kept.tif", "masks/old.tif", "inputs_index.csv"]

        def create_commit(self, repo_id, repo_type, operations, commit_message):
            calls.append(("delete", sorted(op.path_in_repo for op in operations)))

        def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type, **kw):
            calls.append(("upload", path_in_repo, kw.get("delete_patterns")))

        def upload_file(self, **kw):
            calls.append(("file", kw["path_in_repo"]))

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    plan = [(staged, "inputs", True, True)]
    monkeypatch.setattr(ex, "_upload_plan", lambda args: plan)
    ex.upload_to_hf("repo", None)
    assert calls == [("delete", ["inputs/old.tif"]), ("upload", "inputs", None)]   # masks/ untouched, delete first


def test_staged_files_and_index_use_the_row_layout(tmp_path, monkeypatch):
    ex = pytest.importorskip("scripts.export_for_frontend")
    job = tmp_path / "outputs" / "job_0000"
    job.mkdir(parents=True)
    ms = np.random.rand(7, 16, 16).astype("float32")
    mask = np.zeros((16, 16), np.uint8)
    mask[:8, :8] = 1
    save_mask(mask, BBOX, job / "m.tif")
    save_input(ms, BBOX, job / "i.tif")
    entry = {"mask_path": job / "m.tif", "input_path": job / "i.tif", "prob_path": None, "bbox": BBOX,
             "metrics": {"coverage": "25"}, "kind": "flagged", "source": "scan", "nbr_encoding": "raw"}
    work = tmp_path / "final"
    for name, value in (("WORK_DIR", work), ("INPUTS_DIR", work / "inputs"), ("MASKS_DIR", work / "masks"),
                        ("INPUTS_INDEX_PATH", work / "inputs_index.csv")):
        monkeypatch.setattr(ex, name, value)
    results = {"r0341_c0444": entry}
    ex.export_inputs(results)
    ex.export_masks(results)
    ex.export_index(results, "scan-20261009T000000Z")
    assert (work / "inputs/r0341/r0341_c0444.tif").exists() and (work / "masks/r0341/r0341_c0444.tif").exists()
    row = next(csv.DictReader(open(work / "inputs_index.csv")))
    assert (row["input_file"], row["mask_file"]) == ("inputs/r0341/r0341_c0444.tif", "masks/r0341/r0341_c0444.tif")
    assert list(row)[-1] == "scan_id"

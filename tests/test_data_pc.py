"""src/data_pc.py follows src/data.py's GEE recipe — offline: _read is
monkeypatched to serve synthetic scene bands, no Planetary Computer access.

The live check against GEE itself is scripts/compare_sources.py."""
import numpy as np
import pytest
from shapely.geometry import box

from src import data_pc
from src.data_pc import QA_FILL, Scene, composite_tile, mosaic_order
from src.results import is_retryable

SIZE = 4
CLEAR, CLOUD = 1 << 6, (1 << 3) | (1 << 6)   # QA_PIXEL "clear" bit vs. cloud bit set


def scene(sid, cloud):
    return Scene(id=sid, cloud=cloud, day="2024-07-01", geom=box(0, 0, 1, 1),
                 hrefs={k: f"{sid}/{k}" for k in data_pc.PC_BANDS + [data_pc.QA_ASSET]})


def test_ties_go_to_the_later_scene_like_gee():
    a, b, c = scene("LC08_L2SP_226065_20240612", 0.0), scene("LC08_L2SP_226065_20240730", 0.0), scene("x", 5.0)
    assert [s.id for s in mosaic_order([a, c, b])] == [b.id, a.id, c.id]


@pytest.fixture
def two_scenes(monkeypatch):
    """Least cloudy scene: cloud on the left half and fill on one pixel. The
    cloudier scene underneath is clear everywhere. DN 10000 / 20000."""
    qa_top = np.full((SIZE, SIZE), CLEAR, dtype=np.uint16)
    qa_top[:, :2] = CLOUD
    qa_top[3, 3] = QA_FILL
    bands = {
        "top":    {"qa_pixel": qa_top, **{b: np.full((SIZE, SIZE), 10000, np.uint16) for b in data_pc.PC_BANDS}},
        "bottom": {"qa_pixel": np.full((SIZE, SIZE), CLEAR, np.uint16),
                   **{b: np.full((SIZE, SIZE), 20000, np.uint16) for b in data_pc.PC_BANDS}},
    }
    bands["top"]["swir22"] = np.full((SIZE, SIZE), 9000, np.uint16)   # NBR != 0 on top
    reads = []

    def fake_read(href, bbox, size, nodata):
        sid, asset = href.split("/")
        reads.append(href)
        return bands[sid][asset]

    monkeypatch.setattr(data_pc, "_read", fake_read)
    return [scene("top", 1.0), scene("bottom", 30.0)], reads


def test_least_cloudy_valid_pixel_wins_and_gaps_fill_from_below(two_scenes):
    hits, _ = two_scenes
    ms = composite_tile((0, 0, 1, 1), hits, SIZE)
    top, bottom = 10000 * 0.0000275 - 0.2, 20000 * 0.0000275 - 0.2
    assert ms.shape == (7, SIZE, SIZE)
    assert np.allclose(ms[0, :, 2:3], top)              # clear on top -> top scene
    assert np.allclose(ms[0, :, :2], bottom)            # cloud on top -> scene below
    assert np.isclose(ms[0, 3, 3], bottom)              # fill on top -> scene below
    nir, swir2 = top, 9000 * 0.0000275 - 0.2
    assert np.isclose(ms[6, 0, 2], (nir - swir2) / (nir + swir2), atol=1e-6)   # raw NBR, as trained


def test_reading_stops_once_every_pixel_is_filled(monkeypatch):
    reads = []

    def fake_read(href, bbox, size, nodata):
        reads.append(href)
        return np.full((SIZE, SIZE), CLEAR if href.endswith("qa_pixel") else 10000, np.uint16)

    monkeypatch.setattr(data_pc, "_read", fake_read)
    composite_tile((0, 0, 1, 1), [scene("clear", 0.5), scene("below", 30.0)], SIZE)
    assert len(reads) == 7 and all(r.startswith("clear/") for r in reads)   # QA + 6 bands, one scene


def test_pixels_no_scene_covers_stay_nan(monkeypatch):
    monkeypatch.setattr(data_pc, "_read", lambda href, bbox, size, nodata:
                        np.full((SIZE, SIZE), CLOUD if href.endswith("qa_pixel") else 10000, np.uint16))
    ms = composite_tile((0, 0, 1, 1), [scene("s", 1.0)], SIZE)
    assert np.isnan(ms).all()


def test_failure_rows_keep_the_retry_contract():
    # no scene at all is final; a failed download (n_scenes > 0) or a failed
    # scene search (n_scenes "") is retried when the task is resubmitted
    assert not is_retryable({"download_ok": False, "n_scenes": 0})
    assert is_retryable({"download_ok": False, "n_scenes": 3})
    assert is_retryable({"download_ok": False, "n_scenes": ""})


def test_stac_interval_excludes_the_end_date_like_filterdate():
    assert data_pc._datetime_range("2024-06-01", "2024-09-30") == "2024-06-01T00:00:00Z/2024-09-29T23:59:59Z"

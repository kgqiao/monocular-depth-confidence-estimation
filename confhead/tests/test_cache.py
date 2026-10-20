import json

import numpy as np
import pytest
import torch

from confhead.cache.config import CacheConfig
from confhead.cache.store import Cache, CacheMismatch, build_index, done_keys, open_for_build, write_entry
from confhead.dataset import ConfidenceDataset, HeadInputs, RadarConfig, radar_channels
from confhead.geometry import Fit, rasterize
from confhead.labels import LabelConfig, lidar_labels

GRID = (8, 4)
IMAGE = (80, 40)


def points(rows):
    """Rows of (u, v, z, sweep, disparity) -> cached point rows (u, v, z, dt, sweep, disparity)."""
    return np.array([[u, v, z, -0.05 * s, s, d] for u, v, z, s, d in rows], dtype=np.float32).reshape(-1, 6)


def make_entry(scene="scene-a", key="scene-a_000_CAM_FRONT", fit=Fit(0.1, 0.0, 10, 0.0), radar_fit=None):
    # With inverse depth = 0.1 * disparity, disparity 1 means 10 m.
    arrays = {
        "features": np.ones((3, GRID[1], GRID[0]), dtype=np.float16),
        "disparity_grid": np.full((GRID[1], GRID[0]), 2.0, dtype=np.float32),
        "temporal_relstd": np.zeros((GRID[1], GRID[0]), dtype=np.float16),
        "temporal_count": np.ones((GRID[1], GRID[0]), dtype=np.uint8),
        "lidar": points([(5, 5, 10.0, 0, 1.0), (15, 5, 10.0, 0, 0.5), (15, 5, 8.0, 0, 1.25)]),
        "radar": points([(45, 25, 10.0, 0, 1.0), (75, 35, 5.0, 1, 1.0)]),
    }
    meta = {
        "key": key, "scene": scene, "channel": "CAM_FRONT", "index": 0, "condition": "day", "weather": "clear",
        "image": {"width": IMAGE[0], "height": IMAGE[1], "path": "x.jpg"},
        "grid_size": list(GRID), "tap": "output_conv1",
        "fit_lidar": fit.to_dict() if fit else None,
        "fit_radar": (radar_fit or fit).to_dict() if (radar_fit or fit) else None,
        "temporal": {"status": "ok"},
        "lidar": {"points": 3}, "radar": {"points": 2},
    }
    return arrays, meta


def test_rasterize_keeps_the_nearest_point_per_cell():
    pts = points([(15, 5, 10.0, 0, 0), (15, 5, 8.0, 0, 0), (75, 35, 5.0, 0, 0)])
    values, mask = rasterize(pts, np.array([1.0, 2.0, 3.0]), IMAGE, GRID)
    assert mask.sum() == 2
    assert values[0, 1] == 2.0 and values[3, 7] == 3.0


def test_rasterize_radius_fills_the_neighborhood_within_the_grid():
    pts = points([(5, 5, 10.0, 0, 0)])
    _, mask = rasterize(pts, np.array([1.0]), IMAGE, GRID, radius=1)
    assert mask.sum() == 4  # a corner cell and its 3 neighbors inside the grid


def test_lidar_labels_measure_abs_rel_and_threshold_it():
    arrays, meta = make_entry()
    abs_rel, target, mask = lidar_labels(meta, arrays, LabelConfig(tau=0.1))
    # Cell (0, 0): predicted 10 m, lidar 10 m. Cell (0, 1): the 8 m point wins; predicted 8 m.
    assert mask.sum() == 2
    assert abs_rel[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert abs_rel[0, 1] == pytest.approx(0.0, abs=1e-6)
    arrays["lidar"][0, 5] = 0.5  # predicted 20 m against 10 m
    abs_rel, target, mask = lidar_labels(meta, arrays, LabelConfig(tau=0.1))
    assert abs_rel[0, 0] == pytest.approx(1.0) and target[0, 0] == 1.0
    assert target[0, 1] == 0.0


def test_missing_fit_gives_no_labels_and_no_radar():
    arrays, meta = make_entry(fit=None)
    _, _, mask = lidar_labels(meta, arrays)
    residual, valid = radar_channels(meta, arrays)
    assert not mask.any() and not valid.any() and not residual.any()


def test_radar_channels_mark_the_cells_around_each_point():
    arrays, meta = make_entry(radar_fit=Fit(0.05, 0.0, 10, 0.0))  # predicted 20 m against 10 m
    residual, valid = radar_channels(meta, arrays, RadarConfig(radius=1, max_sweeps=1))
    assert valid.sum() == 9
    assert residual[valid > 0] == pytest.approx(1.0)
    _, valid_all = radar_channels(meta, arrays, RadarConfig(radius=0))
    assert valid_all.sum() == 2  # the sweep-1 point too


def test_store_round_trip_index_and_dataset(tmp_path):
    cfg = CacheConfig()
    cfg.output.root = str(tmp_path / "cache")
    cfg.splits = {"train": ["scene-a"], "val": ["scene-b"]}
    root = open_for_build(cfg, overwrite=False)
    for scene in ("scene-a", "scene-b"):
        arrays, meta = make_entry(scene=scene, key=f"{scene}_000_CAM_FRONT")
        meta["fingerprint"] = cfg.fingerprint()
        assert not done_keys(root, [(scene, meta["key"])], cfg.fingerprint())
        write_entry(root, scene, meta["key"], arrays, meta)
        assert done_keys(root, [(scene, meta["key"])], cfg.fingerprint()) == {meta["key"]}
        assert not done_keys(root, [(scene, meta["key"])], "another fingerprint")
    index = build_index(root, cfg.splits)
    assert [r["tags"] for r in index["entries"]] == [["train"], ["val"]]

    cache = Cache(root)
    assert len(cache) == 2 and [e.key for e in cache.entries(tag="val")] == ["scene-b_000_CAM_FRONT"]
    loaded = cache.entry("scene-a_000_CAM_FRONT").load()
    assert loaded["features"].dtype == np.float16 and loaded["lidar"].shape == (3, 6)
    assert json.loads((root / "entries/scene-a/scene-a_000_CAM_FRONT.json").read_text())["arrays"]["lidar"]["shape"] == [3, 6]
    assert cache.entry("scene-a_000_CAM_FRONT").image_path.name == "x.jpg"
    assert Cache(root, dataroot=tmp_path / "data").entry("scene-a_000_CAM_FRONT").image_path == tmp_path / "data/x.jpg"

    ds = ConfidenceDataset(cache, tag="train")
    sample = ds[0]
    assert sample["x"].shape == (3 + 4, GRID[1], GRID[0]) and sample["x"].dtype == torch.float32
    assert sample["target"].shape == sample["mask"].shape == (1, GRID[1], GRID[0])
    assert ds.inputs.channel_names(3)[-4:] == ["depth", "temporal_relstd", "radar_residual", "radar_valid"]
    ablated = ConfidenceDataset(cache, HeadInputs(radar=False, temporal=False), tag="train")
    assert ablated[0]["x"].shape[0] == 3 + 1


def test_a_cache_with_other_settings_is_refused(tmp_path):
    cfg = CacheConfig()
    cfg.output.root = str(tmp_path / "cache")
    open_for_build(cfg, overwrite=False)
    other = CacheConfig()
    other.output.root = cfg.output.root
    other.model.tap = "refinenet1"
    with pytest.raises(CacheMismatch, match="model.tap 'output_conv1' -> 'refinenet1'"):
        open_for_build(other, overwrite=False)
    open_for_build(other, overwrite=True)
    assert json.loads((tmp_path / "cache/cache.json").read_text())["fingerprint"] == other.fingerprint()


def test_stale_entries_leave_the_index(tmp_path):
    cfg = CacheConfig()
    cfg.output.root = str(tmp_path / "cache")
    root = open_for_build(cfg, overwrite=False)
    arrays, meta = make_entry()
    meta["fingerprint"] = "old"
    write_entry(root, "scene-a", meta["key"], arrays, meta)
    index = build_index(root)
    assert index["entries"] == [] and index["stale_entries"] == 1

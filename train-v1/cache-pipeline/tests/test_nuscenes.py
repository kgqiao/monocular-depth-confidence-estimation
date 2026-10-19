"""Tests on the real nuScenes data. They run only when the data (and for the build, the checkpoint) is there."""

import os
from pathlib import Path

import numpy as np
import pytest

from confcache.config import load_config
from confcache.points import load_points

REPO = Path(__file__).resolve().parents[3]
DATAROOT = Path(os.environ.get("NUSCENES_DATAROOT", REPO / "datasets/nuscenes/v1.0-mini"))
CHECKPOINT = os.environ.get("DEPTH_ANYTHING_CHECKPOINT", "")

needs_data = pytest.mark.skipif(not (DATAROOT / "v1.0-mini").is_dir(), reason="no nuScenes v1.0-mini")
needs_model = pytest.mark.skipif(not Path(CHECKPOINT).is_file(), reason="set DEPTH_ANYTHING_CHECKPOINT")


def config(tmp_path, *overrides):
    return load_config(None, [f"source.dataroot={DATAROOT}", f"output.root={tmp_path / 'cache'}", *overrides])


@pytest.fixture(scope="module")
def source():
    from confcache.config import Config
    from confcache.sources import make_source

    cfg = Config()
    cfg.source.dataroot = str(DATAROOT)
    cfg.source.scenes = ["scene-0061"]
    return make_source(cfg)


@needs_data
def test_items_cover_every_camera_of_every_keyframe(source):
    items = source.items()
    assert len(items) == 39 * 6
    assert len({it.key for it in items}) == len(items)
    first, middle = items[0], items[6 * 10]
    assert [o for o, _ in first.neighbors] == [1, 2]  # the first keyframe has no earlier images
    assert [o for o, _ in middle.neighbors] == [-2, -1, 1, 2]
    times = [middle.view.timestamp] + [v.timestamp for _, v in middle.neighbors]
    assert sorted(times) == [times[1], times[2], times[0], times[3], times[4]]
    assert len(middle.lidar) == 1 and 5 <= len(middle.radar) <= 25


@needs_data
def test_lidar_matches_the_confhead_projection(source):
    """The same keyframe lidar pixels as confhead.project, which the overlays checked by eye."""
    from confhead.project import project_lidar

    item = source.items()[6 * 5 + 1]
    pts = load_points(item.lidar, item, 1.0, 80.0)
    nusc = source.nusc
    frame = nusc.frame(item.sample_token)
    ref = project_lidar(nusc, frame, nusc.camera(frame, item.channel))
    mine = np.zeros((item.view.height, item.view.width), dtype=bool)
    mine[np.round(pts[:, 1]).astype(int), np.round(pts[:, 0]).astype(int)] = True
    assert (mine == ref.mask).all()
    assert len(pts) == ref.n_points


@needs_data
@needs_model
def test_build_two_items_and_read_them_back(tmp_path):
    from confcache.build import build
    from confcache.dataset import ConfidenceDataset
    from confcache.report import verify
    from confcache.store import Cache

    cfg = config(tmp_path, f"model.checkpoint={CHECKPOINT}", "source.scenes=[\"scene-0061\"]",
                 "source.cameras=[\"CAM_FRONT\"]", "source.max_samples_per_scene=3", "run.num_workers=0",
                 "temporal.keep_neighbors=true")
    cfg.splits = {"train": ["scene-0061"]}
    report = build(cfg, limit=2, log=lambda m: None)
    assert report.built == 2 and not report.failed

    again = build(cfg, log=lambda m: None)
    assert again.skipped == 2 and again.built == 1

    result = verify(cfg.output.root)
    assert result["entries"] == 3 and result["problems"] == []
    cache = Cache(cfg.output.root)
    meta = cache.entry("scene-0061_001_CAM_FRONT").meta
    assert meta["temporal"]["status"] == "ok" and meta["fit_lidar"]["n"] > 1000
    a = cache.entry("scene-0061_001_CAM_FRONT").load()
    assert a["temporal_aligned"].shape == (4, *a["disparity_grid"].shape)
    sample = ConfidenceDataset(cache, tag="train")[1]
    assert sample["x"].shape[0] == 64 + 4 and sample["mask"].sum() > 1000

    from confcache.report import preview
    assert preview(cache, "scene-0061_001_CAM_FRONT", str(DATAROOT), tmp_path / "previews").stat().st_size > 100_000

"""Tests on the real nuScenes data. They run only when the data (and for the build, the checkpoint) is there."""

import os
from pathlib import Path

import numpy as np
import pytest

from confhead.cache import load_config
from confhead.geometry import apply, pose_to_mat, project
from confhead.items import load_points
from confhead.nuscenes import DATAROOT_ENV, NuScenes, read_lidar

REPO = Path(__file__).resolve().parents[1]
DATAROOT = Path(os.environ.get(DATAROOT_ENV, REPO / "datasets/nuscenes/v1.0-mini"))
CHECKPOINT = os.environ.get("DEPTH_ANYTHING_CHECKPOINT", "")

needs_data = pytest.mark.skipif(not (DATAROOT / "v1.0-mini").is_dir(), reason="no nuScenes v1.0-mini")
needs_model = pytest.mark.skipif(not Path(CHECKPOINT).is_file(), reason="set DEPTH_ANYTHING_CHECKPOINT")


@pytest.fixture(scope="module")
def nusc():
    return NuScenes(DATAROOT)


@needs_data
def test_scenes_carry_condition_and_weather(nusc):
    assert len(nusc.scenes) == 10 and not nusc.skipped_scenes
    assert sum(len(s.samples) for s in nusc.scenes) == 404
    assert nusc.scene("scene-1077").condition == "night"
    assert nusc.scene("scene-0553").condition == "day"
    assert {s.weather for s in nusc.scenes} == {"clear", "rain"}


@needs_data
def test_items_cover_every_camera_of_every_keyframe(nusc):
    items = nusc.items(scenes=["scene-0061"], offsets=(-2, -1, 1, 2), radar_sweeps=5)
    assert len(items) == 39 * 6
    assert len({it.key for it in items}) == len(items)
    first, middle = items[0], items[6 * 10]
    assert [o for o, _ in first.neighbors] == [1, 2]  # the first keyframe has no earlier images
    assert [o for o, _ in middle.neighbors] == [-2, -1, 1, 2]
    times = [middle.view.timestamp] + [v.timestamp for _, v in middle.neighbors]
    assert sorted(times) == [times[1], times[2], times[0], times[3], times[4]]
    assert len(middle.lidar) == 1 and 5 <= len(middle.radar) <= 25
    with pytest.raises(ValueError, match="not in v1.0-mini"):
        nusc.items(scenes=["scene-9999"])


@needs_data
def test_lidar_pixels_match_the_chain_through_the_lidar_ego_frame(nusc):
    """The same pixels as the sensor -> ego -> global -> ego -> camera chain, written out with the tables."""
    item = nusc.items(scenes=["scene-0061"], max_samples_per_scene=6)[6 * 5 + 1]
    pts = load_points(item.lidar, item.view)

    lidar_sd = next(sd for sd in nusc.table["sample_data"] if sd["sample_token"] == item.sample_token
                    and sd["is_key_frame"] and sd["filename"].startswith("samples/LIDAR_TOP/"))
    cam_sd = nusc.get("sample_data", item.view.token)

    def to_global(sd):
        return (pose_to_mat(nusc.get("ego_pose", sd["ego_pose_token"]))
                @ pose_to_mat(nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])))

    raw = read_lidar(DATAROOT / lidar_sd["filename"])
    cam_pts = apply(np.linalg.inv(to_global(cam_sd)) @ to_global(lidar_sd), raw)
    uvz, _ = project(cam_pts, item.view.K, item.view.width, item.view.height, 1.0, 80.0)
    assert len(pts) == len(uvz) > 1000
    assert np.abs(pts[:, :3] - uvz).max() < 1e-3


@needs_data
@needs_model
def test_build_three_items_and_read_them_back(tmp_path):
    from confhead.cache import Cache, build, verify
    from confhead.dataset import ConfidenceDataset
    from confhead.plot import entry_figure

    cfg = load_config(None, [f"source.dataroot={DATAROOT}", f"output.root={tmp_path / 'cache'}",
                             f"model.checkpoint={CHECKPOINT}", "source.scenes=[\"scene-0061\"]",
                             "source.cameras=[\"CAM_FRONT\"]", "source.max_samples_per_scene=3",
                             "run.num_workers=0"])
    cfg.splits = {"train": ["scene-0061"]}
    report = build(cfg, limit=2)
    assert report.built == 2 and not report.failed

    again = build(cfg)
    assert again.skipped == 2 and again.built == 1

    result = verify(cfg.output.root)
    assert result["entries"] == 3 and result["problems"] == []
    cache = Cache(cfg.output.root)
    entry = cache.entry("scene-0061_001_CAM_FRONT")
    assert entry.meta["temporal"]["status"] == "ok" and entry.meta["fit_lidar"]["n"] > 1000
    assert entry.image_path.is_file()
    sample = ConfidenceDataset(cache, tag="train")[1]
    assert sample["x"].shape[0] == 64 + 4 and sample["mask"].sum() > 1000

    out = tmp_path / "figure.png"
    entry_figure(entry).savefig(out, dpi=60)
    assert out.stat().st_size > 50_000

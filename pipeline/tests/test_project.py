import os

import numpy as np
import pytest

from confhead.nuscenes import DATAROOT_ENV, quat_to_mat
from confhead.project import MAX_DEPTH, MIN_DEPTH, project_points

K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 25.0], [0.0, 0.0, 1.0]])
W, H = 100, 50

# Ego frame (x forward, y left, z up) -> camera frame (x right, y down, z forward).
EGO_TO_CAM = np.eye(4)
EGO_TO_CAM[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]


def test_quat_to_mat():
    assert np.allclose(quat_to_mat([1, 0, 0, 0]), np.eye(3))
    s = np.sqrt(0.5)
    yaw90 = quat_to_mat([s, 0, 0, s])
    assert np.allclose(yaw90 @ [1, 0, 0], [0, 1, 0])


def test_point_lands_on_expected_pixel_with_its_depth():
    # 10 m ahead, 1 m right, 0.5 m up -> camera (1, -0.5, 10).
    p = project_points(np.array([[10.0, -1.0, 0.5]]), K, EGO_TO_CAM, W, H)
    assert p.depth.shape == (H, W) and p.depth.dtype == np.float32
    assert p.n_pixels == 1
    assert p.mask[20, 60]
    assert p.depth[20, 60] == pytest.approx(10.0)
    assert p.u[0] == pytest.approx(60.0) and p.v[0] == pytest.approx(20.0)


def test_translation_of_the_pose_is_applied():
    ref_to_cam = EGO_TO_CAM.copy()
    ref_to_cam[:3, 3] = [0.0, 0.0, -2.0]  # camera 2 m ahead of the reference origin
    p = project_points(np.array([[12.0, 0.0, 0.0]]), K, ref_to_cam, W, H)
    assert p.depth[25, 50] == pytest.approx(10.0)


def test_points_behind_outside_or_out_of_range_are_dropped():
    pts = np.array([
        [-10.0, 0.0, 0.0],  # behind the camera
        [10.0, -20.0, 0.0],  # far right, outside the image
        [MIN_DEPTH / 2, 0.0, 0.0],  # too near
        [MAX_DEPTH + 1, 0.0, 0.0],  # too far
    ])
    p = project_points(pts, K, EGO_TO_CAM, W, H)
    assert p.n_points == 0
    assert not p.mask.any()
    assert not p.depth.any()


def test_nearest_point_wins_on_a_shared_pixel():
    pts = np.array([[30.0, 0.0, 0.0], [5.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    p = project_points(pts, K, EGO_TO_CAM, W, H)
    assert p.n_points == 3
    assert p.n_pixels == 1
    assert p.depth[25, 50] == pytest.approx(5.0)
    assert p.fraction == pytest.approx(1 / (W * H))


@pytest.mark.skipif(not os.environ.get(DATAROOT_ENV), reason=f"{DATAROOT_ENV} is not set")
def test_real_keyframe_projects_into_every_camera():
    from confhead.nuscenes import CAMERA_CHANNELS, NuScenes, resolve_dataroot
    from confhead.project import project_lidar, project_radar

    nusc = NuScenes(resolve_dataroot(None))
    scene = nusc.scenes[0]
    frame = nusc.frame(scene["first_sample_token"])
    for channel in CAMERA_CHANNELS:
        cam = nusc.camera(frame, channel)
        lidar = project_lidar(nusc, frame, cam)
        radar = project_radar(nusc, frame, cam)
        assert lidar.depth.shape == (900, 1600)
        assert 0.001 < lidar.fraction < 0.05
        assert radar.n_pixels <= radar.n_points

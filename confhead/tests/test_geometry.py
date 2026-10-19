import numpy as np
import pytest

from confhead.cache.build import with_disparity
from confhead.geometry import (
    apply,
    fit_scale_shift,
    grid_centers,
    project,
    quat_to_mat,
    rescale_coords,
    sample_bilinear,
)
from confhead.items import PointFile, View, fit_to_points, load_points

K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 25.0], [0.0, 0.0, 1.0]])
W, H = 100, 50

# Ego frame (x forward, y left, z up) -> camera frame (x right, y down, z forward).
EGO_TO_CAM = np.eye(4)
EGO_TO_CAM[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]


def test_quat_to_mat():
    assert np.allclose(quat_to_mat([1, 0, 0, 0]), np.eye(3))
    s = np.sqrt(0.5)
    assert np.allclose(quat_to_mat([s, 0, 0, s]) @ [1, 0, 0], [0, 1, 0])  # yaw 90 degrees


def test_project_keeps_points_in_range_and_image():
    pts = np.array([
        [1.0, -0.5, 10.0],  # (60, 20)
        [0.0, 0.0, 0.5],  # too near
        [0.0, 0.0, 90.0],  # too far
        [10.0, 0.0, 10.0],  # right of the image
        [0.0, 0.0, -5.0],  # behind
    ])
    uvz, keep = project(pts, K, W, H, 1.0, 80.0)
    assert keep.tolist() == [True, False, False, False, False]
    assert uvz[0] == pytest.approx([60.0, 20.0, 10.0])


def test_a_point_on_the_last_pixel_edge_stays_inside_after_float32():
    # x = 99.49999999 is inside in float64, but is 99.5 in float32, which rounds to pixel 100.
    x = (99.49999999 - K[0, 2]) / K[0, 0] * 10.0
    uvz, keep = project(np.array([[x, 0.0, 10.0]]), K, W, H, 1.0, 80.0)
    assert not keep[0] or np.round(np.float32(uvz[0, 0])) < W


def test_load_points_moves_each_sweep_into_the_camera(tmp_path):
    ego_to_cam = EGO_TO_CAM.copy()
    ego_to_cam[:3, 3] = [0.0, 0.0, -2.0]  # camera 2 m ahead of the sensor
    view = View("t", tmp_path / "x.jpg", W, H, K, np.eye(4), 0)
    # 12 m ahead, 1 m right, 0.5 m up of the sensor -> camera (1, -0.5, 10) -> pixel (60, 20).
    read = lambda path: np.array([[12.0, -1.0, 0.5], [-5.0, 0.0, 0.0]])
    files = (PointFile(tmp_path / "a", read, ego_to_cam, 0.0, 0), PointFile(tmp_path / "b", read, ego_to_cam, -0.05, 1))
    pts = load_points(files, view)
    assert pts.dtype == np.float32 and pts.shape == (2, 5)
    assert pts[0] == pytest.approx([60.0, 20.0, 10.0, 0.0, 0.0])
    assert pts[1, 3:] == pytest.approx([-0.05, 1.0])
    assert apply(ego_to_cam, np.array([[12.0, -1.0, 0.5]]))[0] == pytest.approx([1.0, -0.5, 10.0])


def test_rescale_maps_pixel_edges_to_edges():
    u, v = rescale_coords(np.array([-0.5, 99.5, 49.5]), np.array([-0.5, 49.5, 24.5]), (100, 50), (10, 5))
    assert u.tolist() == pytest.approx([-0.5, 9.5, 4.5])
    assert v.tolist() == pytest.approx([-0.5, 4.5, 2.0])


def test_grid_centers_cover_the_image_evenly():
    u, v = grid_centers((4, 2), (100, 50))
    assert u.shape == (2, 4)
    assert u[0].tolist() == pytest.approx([12.0, 37.0, 62.0, 87.0])
    assert v[:, 0].tolist() == pytest.approx([12.0, 37.0])


def test_bilinear_is_exact_on_a_plane():
    yy, xx = np.mgrid[0:6, 0:9]
    img = 2.0 * xx + 3.0 * yy + 1.0
    x, y = np.array([0.0, 2.5, 7.25, 8.0]), np.array([0.0, 1.5, 4.75, 5.0])
    assert sample_bilinear(img, x, y) == pytest.approx(2 * x + 3 * y + 1)


def test_fit_recovers_scale_and_shift():
    x = np.linspace(0.5, 10, 50)
    fit = fit_scale_shift(x, 0.02 * x + 0.003, 10)
    assert fit.scale == pytest.approx(0.02) and fit.shift == pytest.approx(0.003)
    assert fit.rmse == pytest.approx(0.0, abs=1e-12)
    assert fit.depth(np.array([5.0]), 1.0, 80.0)[0] == pytest.approx(1 / 0.103)


@pytest.mark.parametrize("x, y", [
    (np.ones(5), np.ones(5)),  # too few
    (np.ones(30), np.arange(30.0)),  # degenerate x
    (np.arange(30.0), -np.arange(30.0)),  # negative scale
])
def test_fit_refuses_what_it_cannot_fit(x, y):
    assert fit_scale_shift(x, y, 10) is None


def test_points_get_the_disparity_at_their_pixel():
    disparity = np.tile(np.arange(10, dtype=np.float32), (5, 1))  # 10 x 5 map of a 100 x 50 image
    pts = np.array([[4.5, 20.0, 10.0, 0.0, 0.0], [94.5, 3.0, 5.0, 0.0, 0.0]], dtype=np.float32)
    out = with_disparity(pts, disparity, (W, H))
    assert out.shape == (2, 6)
    assert out[:, 5] == pytest.approx([0.0, 9.0])
    assert fit_to_points(out, 2) is not None

import numpy as np
import pytest
import torch

from confhead.cache.temporal import temporal_variation
from confhead.geometry import Fit, grid_centers

W, H = 320, 180
K = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 90.0], [0.0, 0.0, 1.0]])
CAM_HEIGHT = 1.5
GRID = (160, 90)


def ground_inverse_depth(size: tuple[int, int], cam_height: float = CAM_HEIGHT) -> np.ndarray:
    """Inverse depth of a flat ground under the camera (y down), 1/80 above the horizon."""
    _, v = grid_centers(size, (W, H))
    below = v - K[1, 2]
    inv = np.where(below > 0, below / (K[1, 1] * cam_height), 0.0)
    return np.clip(inv, 1 / 80, 1.0).astype(np.float32)


def moved(dx: float = 0.0, dz: float = 0.0) -> np.ndarray:
    """key_to_neighbor for a neighbor camera dx right and dz ahead of the keyframe camera."""
    m = np.eye(4)
    m[:3, 3] = [-dx, 0.0, -dz]
    return m


def disparity(inv: np.ndarray, scale: float, shift: float) -> torch.Tensor:
    """A model disparity with its own unknown scale and shift: inv = scale * d + shift."""
    return torch.from_numpy((inv - shift) / scale)


def run(neighbors, key_fit=Fit(0.02, 0.001, 0, 0.0)):
    key = disparity(ground_inverse_depth((W, H)), key_fit.scale, key_fit.shift)
    return temporal_variation(key, GRID, key_fit, K, (W, H), neighbors, 1.0, 80.0, 0.3, 20)


def test_static_ground_with_forward_motion_has_no_spread():
    # Each neighbor sees the same ground from a moved camera, at full resolution and its own scale.
    neighbors = []
    for offset, dz in ((-2, -1.6), (-1, -0.8), (1, 0.8), (2, 1.6)):
        inv = ground_inverse_depth((W, H))  # forward motion over a flat ground: the same image
        neighbors.append((offset, disparity(inv, 0.03 + 0.005 * offset, 0.002), moved(dz=dz), K))
    r = run(neighbors)
    ground = torch.from_numpy(ground_inverse_depth(GRID) > 1 / 80 + 1e-4)
    seen = r.count >= 3
    assert all(n.fit is not None for n in r.neighbors)
    assert seen[ground].float().mean() > 0.7
    # An affine fit to the keyframe disparity, without moving the depth back, gives a median of 0.04 here.
    assert float(r.relstd[ground & seen].median()) < 1e-3
    assert float(r.relstd[ground & seen].quantile(0.95)) < 5e-3
    for n, offset in zip(r.neighbors, (-2, -1, 1, 2)):
        assert n.fit[0] == pytest.approx(0.03 + 0.005 * offset, rel=1e-3)


def test_a_wrong_region_in_one_neighbor_raises_the_spread_there():
    neighbors = []
    for offset, dz in ((-1, -0.8), (1, 0.8)):
        inv = ground_inverse_depth((W, H))
        if offset == 1:
            inv[150:170, 100:160] *= 1.5
        neighbors.append((offset, disparity(inv, 0.03, 0.002), moved(dz=dz), K))
    r = run(neighbors)
    ground = torch.from_numpy(ground_inverse_depth(GRID) > 1 / 80 + 1e-4)
    assert float(r.relstd[ground].median()) < 0.01
    assert float(r.relstd[ground].max()) > 0.1


def test_no_neighbor_fit_leaves_one_frame_and_zero_spread():
    r = run([(1, torch.zeros(H, W), moved(dz=0.8), K)])
    assert r.neighbors[0].fit is None
    assert int(r.count.max()) == 1
    assert float(r.relstd.abs().max()) == 0.0


def test_sideways_motion_leaves_part_of_the_grid_unseen():
    inv = ground_inverse_depth((W, H))
    r = run([(1, disparity(inv, 0.03, 0.002), moved(dx=3.0), K)])
    assert r.neighbors[0].overlap < 1.0
    assert 1 <= int(r.count.min()) and int(r.count.max()) == 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
def test_mps_matches_cpu():
    neighbors = [(1, disparity(ground_inverse_depth((W, H)), 0.03, 0.002), moved(dz=0.8), K)]
    cpu = run(neighbors)
    key = disparity(ground_inverse_depth((W, H)), 0.02, 0.001).to("mps")
    mps = temporal_variation(key, GRID, Fit(0.02, 0.001, 0, 0.0), K, (W, H),
                             [(o, d.to("mps"), m, k) for o, d, m, k in neighbors], 1.0, 80.0, 0.3, 20)
    assert torch.allclose(cpu.relstd, mps.relstd.cpu(), atol=1e-3)

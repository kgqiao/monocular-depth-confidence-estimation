"""Temporal depth variation: how much the model's depth changes across neighbor frames.

For each cell of the keyframe grid:
1. Lift the cell center to 3D with the keyframe depth (disparity through the
   metric fit).
2. Move the point into each neighbor camera with the two camera poses, project
   it, and sample the neighbor disparity there.
3. Fit the sampled neighbor disparity to the inverse depth of the moved point
   (scale and shift, least squares with the worst 10 % of cells dropped),
   because each image has its own unknown scale. This gives the neighbor's own
   metric depth at the sample.
4. Lift the sample to 3D with that depth, move it back into the keyframe
   camera, and keep its inverse depth there.
5. Take std / mean of the inverse depths over the frames that see the cell.

For a static scene and a correct pose, every frame gives the same point, so the
spread is 0 whatever the motion.

The result is a relative spread: 0.05 means the inverse depth varies by about
5 % across frames. Moving objects, occlusions, and wrong keyframe depth all
raise it; that is the signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..geometry import Fit, grid_centers

TRIM = 0.1


@dataclass
class NeighborResult:
    offset: int
    overlap: float  # fraction of grid cells that land inside the neighbor image
    fit: tuple[float, float] | None  # neighbor disparity -> inverse depth, None when it failed


@dataclass
class TemporalResult:
    relstd: torch.Tensor  # (h, w) float32, 0 where fewer than 2 frames see the cell
    count: torch.Tensor  # (h, w) uint8, frames that see the cell, keyframe included
    neighbors: list[NeighborResult]


def _fit(x: torch.Tensor, y: torch.Tensor, min_points: int) -> tuple[float, float] | None:
    """Least squares y = a x + b, then again without the worst TRIM of residuals."""
    for _ in range(2):
        if x.numel() < max(min_points, 2):
            return None
        xm, ym = x.mean(), y.mean()
        var = ((x - xm) ** 2).sum()
        if var <= 1e-12:
            return None
        a = ((x - xm) * (y - ym)).sum() / var
        b = ym - a * xm
        res = (a * x + b - y).abs()
        keep = res <= torch.quantile(res, 1 - TRIM) if res.numel() > 10 else torch.ones_like(res, dtype=torch.bool)
        x, y = x[keep], y[keep]
    a, b = float(a), float(b)
    return (a, b) if a > 0 else None


def _sample(disparity: torch.Tensor, u: torch.Tensor, v: torch.Tensor, image_size: tuple[int, int]):
    """Bilinear samples of a map that covers the whole image, at original pixel coordinates (u, v).
    Returns the samples and whether each (u, v) is inside the image."""
    W, H = image_size
    # grid_sample with align_corners=False maps -1 and 1 to the outer edges of the image.
    gx = (u + 0.5) / W * 2 - 1
    gy = (v + 0.5) / H * 2 - 1
    inside = (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)
    grid = torch.stack([gx, gy], dim=-1).float()[None]
    out = F.grid_sample(disparity[None, None].float(), grid, mode="bilinear",
                        padding_mode="border", align_corners=False)[0, 0]
    return out, inside


def temporal_variation(key_disparity: torch.Tensor, grid_size: tuple[int, int], key_fit: Fit, K: np.ndarray,
                       image_size: tuple[int, int],
                       neighbors: list[tuple[int, torch.Tensor, np.ndarray, np.ndarray]],
                       min_depth: float, max_depth: float, min_overlap: float,
                       min_fit_points: int) -> TemporalResult:
    """
    key_disparity: keyframe disparity at any size; it is sampled at the grid cell centers
        the same way as the neighbors, so that a static scene gives the same value.
    grid_size: (w, h) of the output.
    neighbors: (offset, disparity at any size, key_to_neighbor (4, 4), neighbor K (3, 3)).
    image_size: original (width, height); every map covers the whole image.
    """
    device = key_disparity.device
    w, h = grid_size
    inv_min, inv_max = 1.0 / max_depth, 1.0 / min_depth

    def to_inverse_depth(d: torch.Tensor) -> torch.Tensor:
        return (key_fit.scale * d + key_fit.shift).clamp(inv_min, inv_max)

    dtype = torch.float32 if device.type == "mps" else torch.float64  # MPS has no float64
    u, v = (torch.from_numpy(a).to(dtype).to(device) for a in grid_centers((w, h), image_size))
    key_inv = to_inverse_depth(_sample(key_disparity, u, v, image_size)[0])
    in_range = (key_inv > inv_min) & (key_inv < inv_max)
    z = 1.0 / key_inv.to(dtype)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = torch.stack([(u - cx) / fx * z, (v - cy) / fy * z, z, torch.ones_like(z)], dim=-1)  # (h, w, 4)

    frames = [key_inv]
    seen = [torch.ones_like(key_inv, dtype=torch.bool)]
    results = []
    for offset, nb_disparity, key_to_nb, K_nb in neighbors:
        m = torch.as_tensor(key_to_nb, dtype=dtype, device=device)
        q = pts @ m.T
        zn = q[..., 2]
        ok = zn > min_depth * 0.5
        zs = torch.where(ok, zn, torch.ones_like(zn))
        un = K_nb[0, 0] * q[..., 0] / zs + K_nb[0, 2]
        vn = K_nb[1, 1] * q[..., 1] / zs + K_nb[1, 2]
        sampled, inside = _sample(nb_disparity, un, vn, image_size)
        ok &= inside
        overlap = float(ok.float().mean())
        fit = None
        if overlap >= min_overlap:
            fit_ok = ok & in_range  # depths at a clamp, such as the sky, say nothing about the scale
            fit = _fit(sampled[fit_ok], 1.0 / zs[fit_ok].float(), min_fit_points)
        results.append(NeighborResult(offset=offset, overlap=overlap, fit=fit))
        if fit is None:
            continue
        depth_nb = 1.0 / (fit[0] * sampled + fit[1]).clamp(inv_min, inv_max)
        back = torch.stack([(un - K_nb[0, 2]) / K_nb[0, 0] * depth_nb, (vn - K_nb[1, 2]) / K_nb[1, 1] * depth_nb,
                            depth_nb.to(dtype), torch.ones_like(depth_nb, dtype=dtype)], dim=-1) @ m.inverse().T
        ok &= back[..., 2] > 0
        frames.append((1.0 / back[..., 2].clamp(min=1e-6)).float().clamp(inv_min, inv_max))
        seen.append(ok)

    stack = torch.stack(frames)  # (1 + n, h, w)
    mask = torch.stack(seen).float()
    count = mask.sum(0)
    mean = (stack * mask).sum(0) / count
    var = (((stack - mean) ** 2) * mask).sum(0) / count
    relstd = torch.where(count >= 2, var.sqrt() / mean, torch.zeros_like(mean))
    return TemporalResult(relstd=relstd.float(), count=count.to(torch.uint8), neighbors=results)

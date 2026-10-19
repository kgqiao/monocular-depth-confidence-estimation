"""Build the cache: load, run the frozen model, derive, write.

Worker processes read the images and point files. The main process batches
keyframe and neighbor images through the model on the device, derives the
grid maps, and hands each entry to writer threads. An entry that exists with
the same fingerprint is skipped, so a stopped build continues where it was.
`shard = (i, n)` takes every n-th item from i, for one job of a job array.
"""

from __future__ import annotations

import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .geometry import Fit, fit_to_points, rescale_coords, sample_bilinear
from .model import FrozenDepthAnything, model_size, preprocess
from .points import load_points
from .sources import Item, make_source
from .store import build_index, entry_done, open_for_build, write_entry
from .temporal import temporal_variation


@dataclass
class BuildReport:
    root: Path
    items: int = 0
    built: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0
    images: int = 0


def grid_size(width: int, height: int, cfg: Config) -> tuple[int, int]:
    """(w, h) of the grid for an image of this size."""
    w, h = model_size(width, height, cfg.model.short_side)
    m = cfg.grid.patch_multiple
    return w // 14 * m, h // 14 * m


def select_items(cfg: Config, shard: tuple[int, int] = (0, 1)) -> tuple[list[Item], list[str]]:
    source = make_source(cfg)
    i, n = shard
    return source.items()[i::n], source.skipped_scenes


# ----- loading, in worker processes -------------------------------------


class ItemDataset(Dataset):
    def __init__(self, items: list[Item], cfg: Config):
        self.items = items
        self.short_side = cfg.model.short_side
        self.min_depth = cfg.points.min_depth
        self.max_depth = cfg.points.max_depth

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.items[i]
        try:
            views = [item.view] + [v for _, v in item.neighbors]
            return {
                "item": item,
                "images": np.stack([preprocess(v.image_path, self.short_side) for v in views]),
                "lidar": load_points(item.lidar, item, self.min_depth, self.max_depth),
                "radar": load_points(item.radar, item, self.min_depth, self.max_depth),
            }
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the build
            return {"item": item, "error": f"{type(e).__name__}: {e}"}


def _as_is(sample):
    return sample


# ----- derivation, on the device ----------------------------------------


def resize(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """(C, h, w) -> (C, *size_hw): area average to shrink, bilinear to grow."""
    if tuple(x.shape[-2:]) == tuple(size_hw):
        return x
    shrink = size_hw[0] <= x.shape[-2] and size_hw[1] <= x.shape[-1]
    if not shrink:
        return F.interpolate(x[None], size=size_hw, mode="bilinear", align_corners=False)[0]
    divisible = x.shape[-2] % size_hw[0] == 0 and x.shape[-1] % size_hw[1] == 0
    if x.device.type == "mps" and not divisible:  # MPS pools only by whole factors
        return F.interpolate(x[None].cpu(), size=size_hw, mode="area")[0].to(x.device)
    return F.interpolate(x[None], size=size_hw, mode="area")[0]


def with_disparity(points: np.ndarray, disparity: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Append the model disparity at each point: (N, 5) -> (N, 6)."""
    h, w = disparity.shape
    x, y = rescale_coords(points[:, 0].astype(np.float64), points[:, 1].astype(np.float64), image_size, (w, h))
    d = sample_bilinear(disparity, x, y)
    return np.concatenate([points, d[:, None].astype(np.float32)], axis=1)


def derive(cfg: Config, loaded: dict[str, Any], disparity: torch.Tensor,
           features: torch.Tensor) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """One item's arrays and meta, from its model outputs.

    disparity: (1 + K, hm, wm), keyframe first, then the neighbors in item order.
    features: (C, hf, wf) of the keyframe.
    """
    item: Item = loaded["item"]
    view = item.view
    image_size = (view.width, view.height)
    gw, gh = grid_size(view.width, view.height, cfg)

    key_disp = disparity[0]
    disp_grid = resize(key_disp[None], (gh, gw))[0]
    feat_grid = resize(features, (gh, gw))
    key_disp_np = key_disp.cpu().numpy()

    lidar = with_disparity(loaded["lidar"], key_disp_np, image_size)
    radar = with_disparity(loaded["radar"], key_disp_np, image_size)
    min_points = cfg.temporal.min_fit_points
    fit_lidar = fit_to_points(lidar[lidar[:, 4] == 0], min_points)
    fit_radar = fit_to_points(radar, min_points)

    temporal_meta: dict[str, Any] = {"align": cfg.temporal.align, "enabled": cfg.temporal.enabled}
    relstd = torch.zeros((gh, gw))
    count = torch.ones((gh, gw), dtype=torch.uint8)
    aligned = torch.full((len(cfg.temporal.offsets), gh, gw), float("nan"))
    if cfg.temporal.enabled:
        fit = fit_lidar if cfg.temporal.align == "lidar" else fit_radar
        offsets = [o for o, _ in item.neighbors]
        temporal_meta["missing_offsets"] = [o for o in cfg.temporal.offsets if o not in offsets]
        if fit is None:
            temporal_meta["status"] = f"no {cfg.temporal.align} fit"
        elif not item.neighbors:
            temporal_meta["status"] = "no neighbors"
        else:
            neighbors = [
                (offset, disparity[1 + j], np.linalg.inv(nb.cam_to_global) @ view.cam_to_global, nb.K)
                for j, (offset, nb) in enumerate(item.neighbors)
            ]
            result = temporal_variation(key_disp, (gw, gh), fit, view.K, image_size, neighbors,
                                        cfg.points.min_depth, cfg.points.max_depth,
                                        cfg.temporal.min_overlap, min_points)
            relstd, count = result.relstd, result.count
            for j, offset in enumerate(offsets):
                aligned[cfg.temporal.offsets.index(offset)] = result.aligned[j].cpu()
            temporal_meta["status"] = "ok"
            temporal_meta["neighbors"] = [
                {"offset": r.offset, "token": nb.token, "overlap": round(r.overlap, 4),
                 "fit": None if r.fit is None else {"scale": r.fit[0], "shift": r.fit[1]}}
                for r, (_, nb) in zip(result.neighbors, item.neighbors)
            ]

    feature_dtype = np.float16 if cfg.output.feature_dtype == "float16" else np.float32
    arrays = {
        "features": feat_grid.cpu().numpy().astype(feature_dtype),
        "disparity_grid": disp_grid.cpu().numpy().astype(np.float32),
        "temporal_relstd": relstd.cpu().numpy().astype(np.float16),
        "temporal_count": count.cpu().numpy().astype(np.uint8),
        "lidar": lidar,
        "radar": radar,
    }
    if cfg.output.store_full_disparity:
        arrays["disparity"] = key_disp_np.astype(np.float16)
    if cfg.temporal.keep_neighbors:
        arrays["temporal_aligned"] = aligned.cpu().numpy().astype(np.float16)

    def fit_dict(f: Fit | None):
        return None if f is None else f.to_dict()

    dataroot = Path(cfg.source.dataroot)
    meta = {
        "key": item.key,
        "scene": item.scene,
        "channel": item.channel,
        "index": item.index,
        "sample_token": item.sample_token,
        "condition": item.condition,
        "weather": item.weather,
        "fingerprint": cfg.fingerprint(),
        "image": {
            "token": view.token,
            "path": str(view.image_path.relative_to(dataroot)) if view.image_path.is_relative_to(dataroot)
            else str(view.image_path),
            "width": view.width,
            "height": view.height,
            "timestamp": view.timestamp,
        },
        "K": view.K.tolist(),
        "cam_to_global": view.cam_to_global.tolist(),
        "model_size": [key_disp.shape[1], key_disp.shape[0]],
        "grid_size": [gw, gh],
        "tap": cfg.model.tap,
        "fit_lidar": fit_dict(fit_lidar),
        "fit_radar": fit_dict(fit_radar),
        "temporal": temporal_meta,
        "lidar": {"sweeps": len(item.lidar), "points": int(len(lidar))},
        "radar": {"sweeps": len(item.radar), "points": int(len(radar))},
    }
    return arrays, meta


# ----- the build ------------------------------------------------------------


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def build(cfg: Config, shard: tuple[int, int] = (0, 1), limit: int = 0, overwrite: bool = False,
          log: Callable[[str], None] = _log) -> BuildReport:
    t0 = time.perf_counter()
    items, skipped_scenes = select_items(cfg, shard)
    if skipped_scenes:
        log(f"warning: skipped {len(skipped_scenes)} scenes with missing keyframe files: "
            f"{', '.join(skipped_scenes)}")
    root = open_for_build(cfg, overwrite)
    fingerprint = cfg.fingerprint()
    report = BuildReport(root=root, items=len(items))
    todo = [it for it in items if overwrite or not entry_done(root, it.scene, it.key, fingerprint)]
    report.skipped = len(items) - len(todo)
    if limit:
        todo = todo[:limit]
    log(f"cache {root}: {len(items)} items in this shard, {report.skipped} done, {len(todo)} to build")
    if not todo:
        build_index(root, cfg.splits)
        report.seconds = time.perf_counter() - t0
        return report

    model = FrozenDepthAnything(cfg.model.checkpoint, cfg.model.encoder, cfg.model.tap,
                                cfg.run.device, cfg.run.precision)
    log(f"model {cfg.model.encoder} on {model.device}, {cfg.run.precision}, tap {cfg.model.tap} "
        f"({model.channels} channels)")
    loader = DataLoader(ItemDataset(todo, cfg), batch_size=None, shuffle=False, collate_fn=_as_is,
                        num_workers=cfg.run.num_workers,
                        prefetch_factor=4 if cfg.run.num_workers else None,
                        persistent_workers=False)

    writers = ThreadPoolExecutor(max_workers=cfg.run.writer_threads)
    pending: list[tuple[str, Future]] = []
    t_start = time.perf_counter()

    def drain(limit_pending: int) -> None:
        while len(pending) > limit_pending:
            key, fut = pending.pop(0)
            try:
                fut.result()
                report.built += 1
            except Exception as e:  # noqa: BLE001
                report.failed.append((key, f"write: {type(e).__name__}: {e}"))

    def run_batch(batch: list[dict[str, Any]]) -> None:
        images = torch.from_numpy(np.concatenate([b["images"] for b in batch]))
        disparity, features = model(images)
        start = 0
        for b in batch:
            n = len(b["images"])
            item = b["item"]
            try:
                arrays, meta = derive(cfg, b, disparity[start:start + n], features[start])
                pending.append((item.key, writers.submit(write_entry, root, item.scene, item.key,
                                                         arrays, meta, cfg.output.compress)))
            except Exception as e:  # noqa: BLE001
                if not report.failed:
                    log(traceback.format_exc())
                report.failed.append((item.key, f"derive: {type(e).__name__}: {e}"))
            start += n
            report.images += n
        drain(cfg.run.writer_threads * 4)

    batch: list[dict[str, Any]] = []
    done = 0
    try:
        for loaded in loader:
            done += 1
            if "error" in loaded:
                report.failed.append((loaded["item"].key, f"load: {loaded['error']}"))
                log(f"error: {loaded['item'].key}: {loaded['error']}")
                continue
            n_images = sum(len(b["images"]) for b in batch)
            same_shape = not batch or batch[0]["images"].shape[1:] == loaded["images"].shape[1:]
            if batch and (n_images + len(loaded["images"]) > cfg.run.batch_size or not same_shape):
                run_batch(batch)
                batch = []
            batch.append(loaded)
            if done % cfg.run.log_every == 0:
                rate = done / (time.perf_counter() - t_start)
                eta = (len(todo) - done) / rate if rate else float("inf")
                log(f"{done}/{len(todo)} items  {rate:.2f} items/s  eta {eta / 60:.1f} min")
        if batch:
            run_batch(batch)
    finally:
        drain(0)
        writers.shutdown(wait=True)

    index = build_index(root, cfg.splits)
    report.seconds = time.perf_counter() - t0
    log(f"built {report.built}, skipped {report.skipped}, failed {len(report.failed)} "
        f"in {report.seconds / 60:.1f} min; index holds {len(index['entries'])} entries")
    for key, why in report.failed:
        log(f"failed: {key}: {why}")
    return report

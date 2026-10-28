"""Train the confidence head on a cache: the settings, the dev split, and the run.

`Training(load_train_config(path)).fit()` trains a head and writes its run folder:

    config.json   every setting, the dev scenes, the class weight, the cache, the git commit
    best.pt       the head of the epoch with the lowest dev loss
    last.pt       the head of the last epoch, with the optimizer and the schedule
    history.json  the train and dev loss of each epoch

A run folder that holds last.pt continues after its last epoch, and refuses
other settings. The batch order of an epoch depends only on the seed and the
epoch number, so a continued run sees the same batches as one that never stopped.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import subprocess
import time
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from .cache.config import ALIGN_SOURCES, DEVICES, _coerce, _parse_value, _resolve_path
from .cache.store import Cache
from .dataset import ConfidenceDataset, HeadInputs
from .depth import pick_device
from .head import ConfidenceHead, load_head, masked_bce, save_head

log = logging.getLogger(__name__)

# The Run 1 plan: a positive rate outside this range means the depth needs a scale per camera first.
POSITIVE_RATE_GATE = (0.05, 0.70)


@dataclass
class DataConfig:
    cache: str = ""  # the cache folder, which holds index.json


@dataclass
class HeadConfig:
    widths: list[int] = field(default_factory=lambda: [64, 64, 32])  # the channels of the 3x3 convolutions


@dataclass
class LoopConfig:
    epochs: int = 8
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    samples_per_epoch: int = 0  # train images per epoch, drawn at random; 0: all of them
    dev_fraction: float = 0.1  # the share of the official train scenes that go to dev
    dev_limit: int = 3000  # dev images for the dev loss, a fixed random subset; 0: all of them
    stats_sample: int = 256  # train images for the input statistics and the class weight
    seed: int = 0


@dataclass
class OutputConfig:
    runs: str = "outputs/runs"  # the folder of all runs
    name: str = ""  # the run folder; empty: the cache name and the time


@dataclass
class RunConfig:
    device: str = "auto"
    num_workers: int = -1  # loader processes; -1: one less than the CPUs, at most 8


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    inputs: HeadInputs = field(default_factory=HeadInputs)
    head: HeadConfig = field(default_factory=HeadConfig)
    train: LoopConfig = field(default_factory=LoopConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    run: RunConfig = field(default_factory=RunConfig)


PATH_SETTINGS = (("data", "cache"), ("output", "runs"))
# A continued run must match its config.json in everything but these sections.
RESUMABLE_CHANGES = ("data", "output", "run")


def _build(cls: type, data: Any, name: str) -> Any:
    """`cls` with the defaults, changed by the nested dict `data`; every value checked against its default."""
    if not isinstance(data, dict):
        raise ValueError(f"[{name}] must be a table")
    defaults = cls()
    known = [f.name for f in fields(cls)]
    values = {}
    for key, value in data.items():
        full = f"{name}.{key}" if name else key
        if key not in known:
            what = f"setting {full}" if name else f"section [{key}]"
            raise ValueError(f"unknown {what}; known: {', '.join(known)}")
        default = getattr(defaults, key)
        values[key] = _build(type(default), value, full) if is_dataclass(default) else _coerce(full, value, default)
    return replace(defaults, **values)


def _merge(into: dict, layer: dict) -> None:
    for key, value in layer.items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            _merge(into[key], value)
        else:
            into[key] = value


def _resolve_paths(layer: dict, base: Path) -> None:
    for section, key in PATH_SETTINGS:
        values = layer.get(section)
        if isinstance(values, dict) and isinstance(values.get(key), str):
            values[key] = _resolve_path(values[key], base)


def load_train_config(path: str | Path | None = None, overrides: list[str] | None = None) -> TrainConfig:
    """Defaults, then the TOML file at `path`, then each `section.key=value` of `overrides`.

    Nested settings take more dots: `inputs.labels.tau=0.2`. Paths in the file
    resolve against its folder, and paths in an override against the working folder.
    """
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path).expanduser()
        layer = tomllib.loads(path.read_text())
        _resolve_paths(layer, path.resolve().parent)
        _merge(data, layer)
    for item in overrides or []:
        name, sep, text = item.partition("=")
        keys = name.split(".")
        if not sep or len(keys) < 2 or not all(keys):
            raise ValueError(f"override {item!r}: write it as section.key=value")
        layer = _parse_value(text)
        for key in reversed(keys):
            layer = {key: layer}
        _resolve_paths(layer, Path.cwd())
        _merge(data, layer)
    cfg = _build(TrainConfig, data, "")
    cfg.output.runs = _resolve_path(cfg.output.runs, Path.cwd())
    validate(cfg)
    return cfg


def validate(cfg: TrainConfig) -> None:
    """Fail early, with the setting named, on a value the training cannot use."""
    def check(ok: bool, message: str) -> None:
        if not ok:
            raise ValueError(message)

    i, t = cfg.inputs, cfg.train
    check(bool(cfg.data.cache), "data.cache is empty: set it to the cache folder")
    check(i.features or i.depth or i.temporal or i.radar, "inputs: every channel group is off")
    check(i.labels.align in ALIGN_SOURCES, f"inputs.labels.align {i.labels.align!r}: use lidar or radar")
    check(i.labels.tau > 0, "inputs.labels.tau must be above 0")
    check(i.radar_cfg.align in ALIGN_SOURCES, f"inputs.radar_cfg.align {i.radar_cfg.align!r}: use lidar or radar")
    check(i.radar_cfg.radius >= 0, "inputs.radar_cfg.radius must be 0 or more")
    check(i.radar_cfg.max_sweeps >= 0, "inputs.radar_cfg.max_sweeps must be 0 or more")
    check(all(isinstance(w, int) and w >= 1 for w in cfg.head.widths), "head.widths must hold integers of 1 or more")
    check(t.epochs >= 1, "train.epochs must be 1 or more")
    check(t.batch_size >= 1, "train.batch_size must be 1 or more")
    check(t.lr > 0, "train.lr must be above 0")
    check(t.weight_decay >= 0, "train.weight_decay must be 0 or more")
    check(t.samples_per_epoch >= 0, "train.samples_per_epoch must be 0 or more")
    check(0 < t.dev_fraction < 1, "train.dev_fraction must be in (0, 1)")
    check(t.dev_limit >= 0, "train.dev_limit must be 0 or more")
    check(t.stats_sample >= 1, "train.stats_sample must be 1 or more")
    check(cfg.run.device in DEVICES, f"run.device {cfg.run.device!r}: use one of {', '.join(DEVICES)}")
    check(cfg.run.num_workers >= -1, "run.num_workers must be -1 or more")


def dev_scenes(rows: list[dict[str, Any]], fraction: float, seed: int) -> list[str]:
    """A share of the scenes of each (condition, weather) group; at least one from a group of 2 or more."""
    groups = defaultdict(set)
    for r in rows:
        groups[(r["condition"], r["weather"])].add(r["scene"])
    rng = np.random.default_rng(seed)
    dev = []
    for _, scenes in sorted(groups.items()):
        n = max(1, round(len(scenes) * fraction)) if len(scenes) >= 2 else 0
        dev += rng.choice(sorted(scenes), size=n, replace=False).tolist()
    return sorted(dev)


def git_commit() -> str:
    """The commit of the code, with -dirty when tracked files have changed; empty outside git."""
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = run("rev-parse", "--short", "HEAD")
    return commit + ("-dirty" if commit and run("status", "--porcelain", "--untracked-files=no") else "")


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


@dataclass
class InputStats:
    mean: torch.Tensor  # (C,) per channel of x
    std: torch.Tensor
    images: int
    cells: int
    labeled: int
    positive: int

    @property
    def labeled_fraction(self) -> float:
        return self.labeled / self.cells

    @property
    def positive_rate(self) -> float:
        """The share of labeled cells with an AbsRel above tau."""
        return self.positive / self.labeled

    @property
    def pos_weight(self) -> float:
        """negatives / positives, so the two classes have the same total weight in the loss."""
        return (1 - self.positive_rate) / self.positive_rate

    @property
    def in_gate(self) -> bool:
        return POSITIVE_RATE_GATE[0] <= self.positive_rate <= POSITIVE_RATE_GATE[1]


@dataclass
class Speed:
    load_rate: float  # images per second from the cache
    step_rate: float  # images per second through a training step
    images_per_epoch: int

    @property
    def minutes_per_epoch(self) -> float:
        return self.images_per_epoch / min(self.load_rate, self.step_rate) / 60


class Training:
    """One training run over the official train split of a cache, divided into train and dev by scene.

    `Training(cfg).fit()` does all of it. The other methods are its steps, for a
    notebook to show one by one. `cache` overrides `cfg.data.cache`, for example
    to read the images from another dataroot.
    """

    def __init__(self, cfg: TrainConfig, cache: Cache | None = None):
        validate(cfg)
        self.cfg = cfg
        self.cache = cache or Cache(cfg.data.cache)
        self.device = pick_device(cfg.run.device)
        if cfg.run.num_workers >= 0:
            self.num_workers = cfg.run.num_workers
        else:
            cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
            self.num_workers = max(0, min(8, cpus - 1))
        name = cfg.output.name or f"{self.cache.root.name}-{time.strftime('%Y%m%d-%H%M')}"
        self.dir = Path(cfg.output.runs) / name

        self.dataset = ConfidenceDataset(self.cache, cfg.inputs, tag="train")
        self.channels = self.dataset.channel_names
        rows = self.dataset.rows
        self.dev_scenes = dev_scenes(rows, cfg.train.dev_fraction, cfg.train.seed)
        dev = set(self.dev_scenes)
        self.train_idx = [i for i, r in enumerate(rows) if r["scene"] not in dev]
        self.dev_all = [i for i, r in enumerate(rows) if r["scene"] in dev]
        if not self.train_idx or not self.dev_all:
            raise ValueError(f"{self.cache.root}: the train split gives {len(self.train_idx)} train and "
                             f"{len(self.dev_all)} dev images; dev needs a condition and weather group of 2 or more scenes")
        self.dev_idx = self.dev_all
        if cfg.train.dev_limit and len(self.dev_all) > cfg.train.dev_limit:
            pick = np.random.default_rng(cfg.train.seed).choice(self.dev_all, cfg.train.dev_limit, replace=False)
            self.dev_idx = sorted(pick.tolist())
        self.train_set = Subset(self.dataset, self.train_idx)
        self.dev_set = Subset(self.dataset, self.dev_idx)
        self.stats: InputStats | None = None

    def splits(self) -> list[dict[str, Any]]:
        """One row per split: scenes, images, images used, and night and rain scenes."""
        def row(name: str, idx: list[int], used: int) -> dict[str, Any]:
            rows = [self.dataset.rows[i] for i in idx]
            scenes = lambda **f: len({r["scene"] for r in rows if all(r[k] == v for k, v in f.items())})
            return {"split": name, "scenes": scenes(), "images": len(rows), "used": used,
                    "night_scenes": scenes(condition="night"), "rain_scenes": scenes(weather="rain")}
        return [row("train", self.train_idx, len(self.train_idx)), row("dev", self.dev_all, len(self.dev_idx))]

    def loader(self, dataset, **kwargs) -> DataLoader:
        # With spawn, the default on macOS, each loader process takes 5 s to stop; forkserver stops at once.
        spawn = self.num_workers and mp.get_start_method() == "spawn" and "forkserver" in mp.get_all_start_methods()
        return DataLoader(dataset, batch_size=self.cfg.train.batch_size, num_workers=self.num_workers,
                          pin_memory=self.device.type == "cuda", multiprocessing_context="forkserver" if spawn else None,
                          **kwargs)

    def input_stats(self) -> InputStats:
        """The mean and std of each channel of x, and the label counts, over a random sample of train images."""
        n = min(self.cfg.train.stats_sample, len(self.train_set))
        pick = np.random.default_rng(self.cfg.train.seed).choice(len(self.train_set), n, replace=False)
        C = len(self.channels)
        total, total_sq = torch.zeros(C, dtype=torch.float64), torch.zeros(C, dtype=torch.float64)
        cells = labeled = positive = 0
        for batch in self.loader(Subset(self.train_set, pick.tolist())):
            x = batch["x"].double()
            total += x.sum((0, 2, 3))
            total_sq += (x ** 2).sum((0, 2, 3))
            cells += x.shape[0] * x.shape[2] * x.shape[3]
            labeled += int(batch["mask"].sum().item())
            positive += int((batch["target"] * batch["mask"]).sum().item())
        if not labeled or not positive or positive == labeled:
            raise ValueError(f"{n} train images give {labeled} labeled cells, {positive} of them above tau; "
                             "the class weight needs both classes")
        mean = total / cells
        std = (total_sq / cells - mean ** 2).clamp(min=0).sqrt()
        self.stats = InputStats(mean.float(), std.float(), n, cells, labeled, positive)
        return self.stats

    def new_head(self) -> ConfidenceHead:
        """A head with fresh weights, standardizing x with the input statistics when they are known."""
        head = ConfidenceHead(len(self.channels), tuple(self.cfg.head.widths))
        if self.stats is not None:
            head.set_input_stats(self.stats.mean, self.stats.std)
        return head.to(self.device)

    def speed(self, batches: int = 100, steps: int = 20) -> Speed:
        """Loading and training rates, measured apart; the loading rate includes the start of the loader processes."""
        bs = self.cfg.train.batch_size
        probe = self.loader(self.train_set, shuffle=True, generator=torch.Generator().manual_seed(self.cfg.train.seed))
        n = min(batches, len(probe))
        t0 = time.perf_counter()
        for i, batch in enumerate(probe, 1):
            if i == n:
                break
        load_rate = n * bs / (time.perf_counter() - t0)

        trial = self.new_head()
        optimizer = torch.optim.AdamW(trial.parameters())
        x, target, mask = (batch[k].to(self.device) for k in ("x", "target", "mask"))
        pos_weight = self.stats.pos_weight if self.stats else None
        for i in range(steps + 1):
            if i == 1:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            optimizer.zero_grad()
            loss = masked_bce(trial(x), target, mask, pos_weight)
            loss.backward()
            optimizer.step()
            loss.item()  # waits for the device
        step_rate = steps * bs / (time.perf_counter() - t0)
        return Speed(load_rate, step_rate, self.cfg.train.samples_per_epoch or len(self.train_set))

    @torch.no_grad()
    def dev_loss(self, head: ConfidenceHead, loader: DataLoader, pos_weight: float) -> float:
        """The masked loss over every labeled cell of the loader."""
        head.eval()
        total = labeled = 0.0
        for batch in loader:
            target, mask = batch["target"].to(self.device), batch["mask"].to(self.device)
            n = mask.sum().item()
            logits = head(batch["x"].to(self.device, non_blocking=True))
            total += masked_bce(logits, target, mask, pos_weight).item() * n
            labeled += n
        return total / max(labeled, 1)

    def _config(self, pos_weight: float) -> dict[str, Any]:
        stats = self.stats
        return {
            "settings": asdict(self.cfg), "channels": self.channels, "cache": self._cache_info(),
            "dev_scenes": self.dev_scenes, "dev_used": len(self.dev_idx), "train_images": len(self.train_idx),
            "positive_rate": stats.positive_rate, "pos_weight": pos_weight, "stats_sample": stats.images,
            "git": git_commit(), "torch": torch.__version__, "device": str(self.device),
        }

    def _cache_info(self) -> dict[str, Any]:
        meta = self.cache.entry(self.dataset.rows[0]).meta
        return {"root": str(self.cache.root), "fingerprint": self.cache.header["fingerprint"],
                "grid": meta["grid_size"], "model": self.cache.settings["model"]}

    def _check_same_run(self) -> None:
        """Refuse to continue a run folder with other settings or another cache build."""
        path = self.dir / "config.json"
        if not path.is_file():
            raise ValueError(f"{self.dir} holds last.pt but no config.json; use another output.name")
        saved = json.loads(path.read_text())
        old = {k: v for k, v in _flat(saved["settings"]).items() if k.split(".")[0] not in RESUMABLE_CHANGES}
        new = {k: v for k, v in _flat(asdict(self.cfg)).items() if k.split(".")[0] not in RESUMABLE_CHANGES}
        changes = [f"{k} {old.get(k)!r} -> {new.get(k)!r}" for k in sorted(old.keys() | new.keys())
                   if old.get(k) != new.get(k)]
        if saved["cache"]["fingerprint"] != self.cache.header["fingerprint"]:
            changes.append(f"cache fingerprint {saved['cache']['fingerprint']} -> {self.cache.header['fingerprint']}")
        if changes:
            raise ValueError(f"{self.dir} is a run with other settings: {'; '.join(changes)}. "
                             "Use another output.name for a new run")

    def fit(self) -> list[dict[str, Any]]:
        """Train to `train.epochs`, after the last epoch of the run folder's last.pt if it has one.

        Returns the history: one dict per epoch with its train and dev loss.
        """
        cfg, device = self.cfg, self.device
        self.dir.mkdir(parents=True, exist_ok=True)
        last, best = self.dir / "last.pt", self.dir / "best.pt"

        generator = torch.Generator()
        sampler = RandomSampler(self.train_set, num_samples=cfg.train.samples_per_epoch or len(self.train_set),
                                generator=generator)
        train_loader = self.loader(self.train_set, sampler=sampler, drop_last=True,
                                   persistent_workers=self.num_workers > 0)
        if not len(train_loader):
            raise ValueError(f"{len(sampler)} train images make no batch of {cfg.train.batch_size}")
        dev_loader = self.loader(self.dev_set)

        state = None
        if last.is_file():
            self._check_same_run()
            head, _, state = load_head(last, device)
            history, pos_weight = state["history"], state["pos_weight"]
        else:
            if self.stats is None:
                self.input_stats()
            # After the statistics, whose loader draws from the global RNG, so the weights do not depend on who ran them.
            torch.manual_seed(cfg.train.seed)
            head = self.new_head()
            history, pos_weight = [], self.stats.pos_weight
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs * len(train_loader))
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if len(history) >= cfg.train.epochs:
                log.info("%s already holds %d of %d epochs; nothing to train", self.dir, len(history), cfg.train.epochs)
            else:
                log.info("continuing %s after epoch %d of %d", self.dir, len(history), cfg.train.epochs)
        else:
            config = self._config(pos_weight)
            (self.dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")
            log.info("run %s: %d train images, %d dev images, %d steps per epoch, device %s, %d loader processes",
                     self.dir, len(sampler), len(self.dev_idx), len(train_loader), device, self.num_workers)
            log.info("positive rate %.1f%%, class weight %.3f", 100 * self.stats.positive_rate, pos_weight)
        best_loss = min((h["dev_loss"] for h in history), default=float("inf"))
        extra = {"pos_weight": pos_weight, "cache": self._cache_info(), "git": git_commit()}

        log_every = max(1, len(train_loader) // 5)
        for epoch in range(len(history), cfg.train.epochs):
            head.train()
            generator.manual_seed(cfg.train.seed + epoch)
            losses, t0 = [], time.perf_counter()
            for step, batch in enumerate(train_loader, 1):
                x, target, mask = (batch[k].to(device, non_blocking=True) for k in ("x", "target", "mask"))
                loss = masked_bce(head(x), target, mask, pos_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
                if step % log_every == 0:
                    rate = step * cfg.train.batch_size / (time.perf_counter() - t0)
                    eta = (len(train_loader) - step) * cfg.train.batch_size / rate / 60
                    log.info("epoch %d step %d/%d: loss %.4f, %.0f images/s, eta %.1f min", epoch + 1, step,
                             len(train_loader), np.mean(losses[-log_every:]), rate, eta)

            record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                      "dev_loss": self.dev_loss(head, dev_loader, pos_weight),
                      "lr": scheduler.get_last_lr()[0], "minutes": (time.perf_counter() - t0) / 60}
            history.append(record)
            is_best = record["dev_loss"] < best_loss
            if is_best:
                best_loss = record["dev_loss"]
                save_head(best, head, cfg.inputs, self.channels, epoch=epoch + 1, history=history, **extra)
            save_head(last, head, cfg.inputs, self.channels, epoch=epoch + 1, history=history,
                      optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), **extra)
            (self.dir / "history.json").write_text(json.dumps(history, indent=1) + "\n")
            log.info("epoch %d: train loss %.4f, dev loss %.4f, %.1f min%s", epoch + 1, record["train_loss"],
                     record["dev_loss"], record["minutes"], ", best so far" if is_best else "")
        return history

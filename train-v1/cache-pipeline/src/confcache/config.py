"""Build settings.

Every setting has a default here. A TOML file changes some of them, and
`--set section.key=value` on the command line changes them again. Relative
paths in a file resolve against the folder of that file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from confhead.nuscenes import CAMERA_CHANNELS

DATAROOT_ENV = "NUSCENES_DATAROOT"
CHECKPOINT_ENV = "DEPTH_ANYTHING_CHECKPOINT"

TAPS = ("refinenet2", "refinenet1", "output_conv1")
ALIGN_SOURCES = ("lidar", "radar")
PRECISIONS = ("fp32", "fp16", "bf16")
DEVICES = ("auto", "cuda", "mps", "cpu")


@dataclass
class SourceConfig:
    kind: str = "nuscenes"
    dataroot: str = ""  # empty: $NUSCENES_DATAROOT
    version: str = "v1.0-mini"
    cameras: list[str] = field(default_factory=lambda: list(CAMERA_CHANNELS))
    scenes: list[str] = field(default_factory=list)  # empty: every scene on disk
    max_samples_per_scene: int = 0  # 0: every keyframe


@dataclass
class ModelConfig:
    checkpoint: str = ""  # empty: $DEPTH_ANYTHING_CHECKPOINT
    encoder: str = "vitb"
    short_side: int = 518
    tap: str = "output_conv1"


@dataclass
class GridConfig:
    # The dense maps share one grid: the ViT patch grid times this factor.
    patch_multiple: int = 4


@dataclass
class TemporalConfig:
    enabled: bool = True
    offsets: list[int] = field(default_factory=lambda: [-2, -1, 1, 2])
    align: str = "lidar"
    min_fit_points: int = 20
    min_overlap: float = 0.3
    keep_neighbors: bool = False


@dataclass
class PointsConfig:
    min_depth: float = 1.0
    max_depth: float = 80.0
    lidar_sweeps: int = 1
    radar_sweeps: int = 5


@dataclass
class OutputConfig:
    root: str = "outputs/cache/nuscenes-mini"
    feature_dtype: str = "float16"
    store_full_disparity: bool = True
    compress: bool = False


@dataclass
class RunConfig:
    device: str = "auto"
    precision: str = "fp32"
    batch_size: int = 10
    num_workers: int = 4
    writer_threads: int = 2
    log_every: int = 25


@dataclass
class Config:
    source: SourceConfig = field(default_factory=SourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    points: PointsConfig = field(default_factory=PointsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    run: RunConfig = field(default_factory=RunConfig)
    # Scene tags such as train and val, by scene name. They go into the index only.
    splits: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content(self) -> dict[str, Any]:
        """The settings that change what an entry holds, flattened to `section.key`.

        Which scenes and cameras, the paths, the splits, and the run speed
        settings are left out: they change which entries exist or how fast,
        not what an entry holds. The checkpoint counts by its file name.
        """
        d = self.to_dict()
        content = {
            "source.kind": d["source"]["kind"],
            "source.version": d["source"]["version"],
            "model.checkpoint": Path(self.model.checkpoint).name,
            "run.precision": d["run"]["precision"],
            "output.feature_dtype": d["output"]["feature_dtype"],
            "output.store_full_disparity": d["output"]["store_full_disparity"],
        }
        for section in ("model", "grid", "temporal", "points"):
            for k, v in d[section].items():
                content.setdefault(f"{section}.{k}", v)
        return dict(sorted(content.items()))

    def fingerprint(self) -> str:
        blob = json.dumps(self.content(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _parse_value(text: str) -> Any:
    """A TOML value, else the text as a string: `5`, `true`, `[-1, 1]`, `"x"`, x."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def _coerce(name: str, value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false, not {value!r}")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, not {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be a number, not {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string, not {value!r}")
        return value
    if isinstance(default, list):
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a list, not {value!r}")
        return value
    return value


def _apply(cfg: Config, data: dict[str, Any], origin: str) -> None:
    for section, values in data.items():
        if section == "splits":
            if not isinstance(values, dict):
                raise ValueError(f"{origin}: [splits] must map a tag to a list of scene names")
            cfg.splits = {tag: list(names) for tag, names in values.items()}
            continue
        target = getattr(cfg, section, None)
        if not is_dataclass(target):
            raise ValueError(f"{origin}: unknown section [{section}]")
        if not isinstance(values, dict):
            raise ValueError(f"{origin}: [{section}] must be a table")
        known = {f.name for f in fields(target)}
        for key, value in values.items():
            if key not in known:
                raise ValueError(f"{origin}: unknown setting {section}.{key}; "
                                 f"known: {', '.join(sorted(known))}")
            setattr(target, key, _coerce(f"{section}.{key}", value, getattr(target, key)))


def _resolve_path(value: str, base: Path) -> str:
    if not value:
        return value
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Defaults, then the TOML file at `path`, then each `section.key=value`."""
    cfg = Config()
    base = Path.cwd()
    if path is not None:
        path = Path(path).expanduser()
        base = path.resolve().parent
        _apply(cfg, tomllib.loads(path.read_text()), str(path))
        cfg.source.dataroot = _resolve_path(cfg.source.dataroot, base)
        cfg.model.checkpoint = _resolve_path(cfg.model.checkpoint, base)
        cfg.output.root = _resolve_path(cfg.output.root, base)

    for item in overrides or []:
        name, sep, text = item.partition("=")
        if not sep or "." not in name:
            raise ValueError(f"--set {item!r}: write it as section.key=value")
        section, key = name.split(".", 1)
        _apply(cfg, {section: {key: _parse_value(text)}}, "--set")
        if (section, key) in {("source", "dataroot"), ("model", "checkpoint"), ("output", "root")}:
            obj = getattr(cfg, section)
            setattr(obj, key, _resolve_path(getattr(obj, key), Path.cwd()))

    if not cfg.source.dataroot:
        cfg.source.dataroot = _resolve_path(os.environ.get(DATAROOT_ENV, ""), Path.cwd())
    if not cfg.model.checkpoint:
        cfg.model.checkpoint = _resolve_path(os.environ.get(CHECKPOINT_ENV, ""), Path.cwd())
    cfg.output.root = _resolve_path(cfg.output.root, base)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Fail early, with the setting named, on a value the build cannot use."""
    def check(ok: bool, message: str) -> None:
        if not ok:
            raise ValueError(message)

    check(cfg.source.kind in ("nuscenes", "stf"), f"source.kind {cfg.source.kind!r}: use nuscenes or stf")
    unknown = set(cfg.source.cameras) - set(CAMERA_CHANNELS) if cfg.source.kind == "nuscenes" else set()
    check(not unknown, f"source.cameras: unknown channels {sorted(unknown)}")
    check(len(cfg.source.cameras) > 0, "source.cameras is empty")
    check(cfg.source.max_samples_per_scene >= 0, "source.max_samples_per_scene must be 0 or more")
    check(cfg.model.encoder in ("vits", "vitb", "vitl"), f"model.encoder {cfg.model.encoder!r}: use vits, vitb or vitl")
    check(cfg.model.short_side >= 14 and cfg.model.short_side % 14 == 0,
          "model.short_side must be a multiple of the 14 px patch")
    check(cfg.model.tap in TAPS, f"model.tap {cfg.model.tap!r}: use one of {', '.join(TAPS)}")
    check(cfg.grid.patch_multiple in (1, 2, 4, 8, 14), "grid.patch_multiple must be 1, 2, 4, 8 or 14")
    check(all(o != 0 for o in cfg.temporal.offsets), "temporal.offsets must not hold 0, the keyframe itself")
    check(len(set(cfg.temporal.offsets)) == len(cfg.temporal.offsets), "temporal.offsets has duplicates")
    check(cfg.temporal.align in ALIGN_SOURCES, f"temporal.align {cfg.temporal.align!r}: use lidar or radar")
    check(cfg.temporal.min_fit_points >= 2, "temporal.min_fit_points must be 2 or more")
    check(0.0 < cfg.temporal.min_overlap <= 1.0, "temporal.min_overlap must be in (0, 1]")
    check(0.0 < cfg.points.min_depth < cfg.points.max_depth, "points: need 0 < min_depth < max_depth")
    check(cfg.points.lidar_sweeps >= 1 and cfg.points.radar_sweeps >= 1, "points: sweeps must be 1 or more")
    check(cfg.output.feature_dtype in ("float16", "float32"), "output.feature_dtype: use float16 or float32")
    check(cfg.run.device in DEVICES, f"run.device {cfg.run.device!r}: use one of {', '.join(DEVICES)}")
    check(cfg.run.precision in PRECISIONS, f"run.precision {cfg.run.precision!r}: use one of {', '.join(PRECISIONS)}")
    check(cfg.run.batch_size >= 1, "run.batch_size must be 1 or more")
    check(cfg.run.num_workers >= 0 and cfg.run.writer_threads >= 1, "run: num_workers >= 0, writer_threads >= 1")
    tagged = [s for names in cfg.splits.values() for s in names]
    dupes = sorted({s for s in tagged if tagged.count(s) > 1})
    check(not dupes, f"splits: scenes in more than one tag: {dupes}")


def to_toml(cfg: Config) -> str:
    """The settings as TOML, to keep beside the cache and to read back."""
    def value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        return repr(v)

    d = cfg.to_dict()
    lines = []
    for section, values in d.items():
        lines.append(f"[{section}]")
        lines.extend(f"{k} = {value(v)}" for k, v in values.items())
        lines.append("")
    return "\n".join(lines)

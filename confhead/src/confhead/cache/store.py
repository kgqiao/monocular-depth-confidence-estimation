"""The cache on disk.

    <root>/
      cache.json                 settings, fingerprint, and versions of the build
      config.toml                the same settings as TOML, to build again with
      index.json                 one row per entry, with its scene tags
      entries/<scene>/<key>.npz  the arrays of one camera image
      entries/<scene>/<key>.json what the arrays mean; written last, so it marks a complete entry

Arrays of an entry (h, w: the grid):
    features          (C, h, w)   float16   decoder features at the tap
    disparity_grid    (h, w)      float32   model disparity on the grid
    temporal_relstd   (h, w)      float16   see confhead.cache.temporal
    temporal_count    (h, w)      uint8     frames that see the cell, keyframe included
    lidar             (N, 6)      float32   u, v, z, dt, sweep, disparity; see confhead.items
    radar             (M, 6)      float32   the same for radar
"""

from __future__ import annotations

import io
import json
import os
import platform
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry import Fit
from .config import CacheConfig, to_toml

FORMAT_VERSION = 1


class CacheMismatch(RuntimeError):
    """The cache folder holds entries built with other settings."""


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1) + "\n").encode()


def entry_paths(root: Path, scene: str, key: str) -> tuple[Path, Path]:
    d = Path(root) / "entries" / scene
    return d / f"{key}.npz", d / f"{key}.json"


def write_entry(root: Path, scene: str, key: str, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    npz, js = entry_paths(root, scene, key)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    _atomic_write(npz, buf.getvalue())
    meta = {**meta, "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}}
    _atomic_write(js, _json_bytes(meta))


def done_keys(root: Path, entries: Iterable[tuple[str, str]], fingerprint: str) -> set[str]:
    """The keys of the (scene, key) entries that are complete and carry this fingerprint.

    Lists each scene folder once, and reads only the meta of the entries in it.
    """
    listings: dict[str, set[str]] = {}
    done = set()
    for scene, key in entries:
        if scene not in listings:
            try:
                listings[scene] = set(os.listdir(Path(root) / "entries" / scene))
            except FileNotFoundError:
                listings[scene] = set()
        if not {f"{key}.npz", f"{key}.json"} <= listings[scene]:
            continue
        try:
            if json.loads(entry_paths(root, scene, key)[1].read_text()).get("fingerprint") == fingerprint:
                done.add(key)
        except (OSError, json.JSONDecodeError):
            pass
    return done


def _versions() -> dict[str, str]:
    import torch
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}


def open_for_build(cfg: CacheConfig, overwrite: bool) -> Path:
    """Make the cache folder, or check that the one there was built with the same settings.

    With overwrite, a cache built with other settings is kept on disk but its
    header is replaced, so every entry is built again.
    """
    root = Path(cfg.output.root)
    header = root / "cache.json"
    fingerprint = cfg.fingerprint()
    if header.is_file():
        old = json.loads(header.read_text())
        if old.get("fingerprint") != fingerprint and not overwrite:
            new, was = cfg.content(), old.get("content", {})
            diff = [f"{k} {was.get(k)!r} -> {new.get(k)!r}"
                    for k in sorted(set(new) | set(was)) if new.get(k) != was.get(k)]
            raise CacheMismatch(
                f"{root} holds a cache built with other settings: {'; '.join(diff) or 'see cache.json'}. "
                "Set another output.root, or build with overwrite=True to build every entry again.")
        if old.get("fingerprint") == fingerprint:
            return root
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write(header, _json_bytes({
        "format_version": FORMAT_VERSION,
        "fingerprint": fingerprint,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": _versions(),
        "content": cfg.content(),
        "settings": cfg.to_dict(),
    }))
    _atomic_write(root / "config.toml", to_toml(cfg).encode())
    return root


def tags_of(scene: str, splits: dict[str, list[str]]) -> list[str]:
    return [tag for tag, names in splits.items() if scene in names]


def build_index(root: Path, splits: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Scan the complete entries of the current fingerprint into index.json.

    Splits default to the ones the cache was built with. Entries of another
    fingerprint are left out and counted.
    """
    root = Path(root)
    header = json.loads((root / "cache.json").read_text())
    fingerprint = header["fingerprint"]
    if splits is None:
        splits = header["settings"].get("splits", {})
    rows, stale = [], 0
    for js in sorted((root / "entries").glob("*/*.json")):
        meta = json.loads(js.read_text())
        if meta.get("fingerprint") != fingerprint or not js.with_suffix(".npz").is_file():
            stale += 1
            continue
        rows.append({
            "key": meta["key"],
            "scene": meta["scene"],
            "channel": meta["channel"],
            "index": meta["index"],
            "condition": meta["condition"],
            "weather": meta["weather"],
            "tags": tags_of(meta["scene"], splits),
            "path": str(js.with_suffix(".npz").relative_to(root)),
        })
    index = {"fingerprint": fingerprint, "splits": splits, "stale_entries": stale, "entries": rows}
    _atomic_write(root / "index.json", _json_bytes(index))
    return index


@dataclass
class Entry:
    meta: dict[str, Any]
    path: Path
    dataroot: Path  # the dataset folder the cache was built from

    def load(self, names: list[str] | None = None) -> dict[str, np.ndarray]:
        with np.load(self.path) as z:
            return {k: z[k] for k in (names or z.files)}

    def image(self) -> np.ndarray:
        """The camera image from the dataset: (height, width, 3) uint8 RGB."""
        from PIL import Image
        if not self.image_path.is_file():
            raise FileNotFoundError(f"{self.image_path}: the dataset is not at {self.dataroot}; "
                                    "open the cache with Cache(root, dataroot=...) to say where it is")
        return np.asarray(Image.open(self.image_path).convert("RGB"))

    def fit(self, align: str = "lidar") -> Fit | None:
        """The scale-and-shift fit that makes the disparity metric, "lidar" or "radar"; None when it failed."""
        return Fit.from_dict(self.meta[f"fit_{align}"])

    @property
    def key(self) -> str:
        return self.meta["key"]

    @property
    def image_path(self) -> Path:
        return self.dataroot / self.meta["image"]["path"]


class Cache:
    """Read a built cache: `Cache(root).entries(tag="train")`.

    dataroot: where the dataset is on this machine; default the folder the cache was built from.
    """

    def __init__(self, root: str | Path, dataroot: str | Path | None = None):
        self.root = Path(root)
        if not (self.root / "index.json").is_file():
            raise FileNotFoundError(f"{self.root} has no index.json; call confhead.cache.build_index on it.")
        self.header = json.loads((self.root / "cache.json").read_text())
        self.index = json.loads((self.root / "index.json").read_text())
        if self.index["fingerprint"] != self.header["fingerprint"]:
            raise RuntimeError(f"{self.root}/index.json is out of date; call confhead.cache.build_index on it.")
        self.dataroot = Path(dataroot or self.header["settings"]["source"]["dataroot"])
        self._by_key = {r["key"]: r for r in self.index["entries"]}

    @property
    def settings(self) -> dict[str, Any]:
        return self.header["settings"]

    def rows(self, tag: str | None = None, scene: str | None = None, channel: str | None = None,
             condition: str | None = None, weather: str | None = None) -> list[dict[str, Any]]:
        def keep(r: dict) -> bool:
            return ((tag is None or tag in r["tags"]) and (scene is None or r["scene"] == scene)
                    and (channel is None or r["channel"] == channel)
                    and (condition is None or r["condition"] == condition)
                    and (weather is None or r["weather"] == weather))
        return [r for r in self.index["entries"] if keep(r)]

    def entry(self, row_or_key: dict | str) -> Entry:
        row = row_or_key if isinstance(row_or_key, dict) else self._row(row_or_key)
        npz = self.root / row["path"]
        return Entry(meta=json.loads(npz.with_suffix(".json").read_text()), path=npz, dataroot=self.dataroot)

    def entries(self, **filters) -> list[Entry]:
        return [self.entry(r) for r in self.rows(**filters)]

    def _row(self, key: str) -> dict:
        if key not in self._by_key:
            raise KeyError(f"no entry {key!r} in {self.root}")
        return self._by_key[key]

    def __len__(self) -> int:
        return len(self.index["entries"])

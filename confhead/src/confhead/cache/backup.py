"""Back up a cache to a Hugging Face dataset repository, and restore it.

    <repo>/
      README.md                          the dataset card, written when the repository has none
      <cache>/manifest.json              the shards, and the files of the cache each one holds
      <cache>/cache.json, config.toml   the header, sent with each shard
      <cache>/index.json                sent when every shard is up
      <cache>/shards/<00000>.tar.zst     entries/<scene>/ of `scenes_per_shard` scenes, in name order

<cache> is the name of the cache folder, so one repository holds several caches.
A shard is a plain tar compressed with zstd. Any tar with zstd extracts it into
a cache folder: `tar --zstd -xf 00000.tar.zst -C <root>`.

A shard's content key is a hash of the path and size of each file it holds.
Upload skips a shard whose key matches the manifest in the repository, so a
stopped upload continues, and a later upload sends only the shards that changed.
Each shard goes up in one commit with the manifest, so the manifest always
describes the shards in the repository.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .store import build_index

log = logging.getLogger("confhead.cache")

MANIFEST_VERSION = 1
REPO_TYPE = "dataset"
HEADER_FILES = ("cache.json", "config.toml")
READ_CHUNK = 1 << 24
TYPE_NAMES = {str: "a string", int: "an integer", bool: "true or false"}


@dataclass
class BackupConfig:
    repo_id: str = ""  # <user or org>/<name> on huggingface.co
    private: bool = False
    scenes_per_shard: int = 25
    level: int = 3  # zstd level; higher levels compress the float16 features no better
    staging: str = ""  # where a shard waits on disk; empty: <cache>.staging beside the cache


def load_backup_config(path: str | Path) -> BackupConfig:
    """The [huggingface] and [shards] tables of a TOML file; see configs/huggingface.toml."""
    path = Path(path).expanduser()
    data = tomllib.loads(path.read_text())
    cfg = BackupConfig()
    known = {f.name: f for f in fields(cfg)}
    for section, values in data.items():
        if section not in ("huggingface", "shards") or not isinstance(values, dict):
            raise ValueError(f"{path}: unknown section [{section}]; use [huggingface] and [shards]")
        for key, value in values.items():
            if key not in known:
                raise ValueError(f"{path}: unknown setting {section}.{key}; known: {', '.join(sorted(known))}")
            kind = type(getattr(cfg, key))
            if type(value) is not kind:
                raise ValueError(f"{path}: {section}.{key} must be {TYPE_NAMES[kind]}, not {value!r}")
            setattr(cfg, key, value)
    if cfg.staging:
        p = Path(os.path.expandvars(cfg.staging)).expanduser()
        cfg.staging = str(p if p.is_absolute() else (path.resolve().parent / p).resolve())
    if cfg.scenes_per_shard < 1:
        raise ValueError(f"{path}: shards.scenes_per_shard must be 1 or more")
    if not 1 <= cfg.level <= 22:
        raise ValueError(f"{path}: shards.level must be in 1 to 22")
    return cfg


def _require_repo(cfg: BackupConfig) -> str:
    if cfg.repo_id.count("/") != 1:
        raise ValueError("Set huggingface.repo_id in the backup config to <user or org>/<name>, "
                         f"not {cfg.repo_id!r}")
    return cfg.repo_id


# ----- shards -----------------------------------------------------------------


@dataclass
class Shard:
    name: str
    scenes: list[str]
    files: int
    bytes: int
    content: str
    archive_bytes: int = 0
    sha256: str = ""
    paths: list[str] = field(default_factory=list, repr=False)  # relative to the cache root

    def record(self) -> dict[str, Any]:
        d = asdict(self)
        del d["paths"]
        return d


def _scene_files(root: Path, scene: str) -> list[tuple[str, int]]:
    """(path relative to root, size) of each file of a scene, without unfinished temp files."""
    try:
        listing = list(os.scandir(root / "entries" / scene))
    except FileNotFoundError:
        return []
    return sorted((f"entries/{scene}/{f.name}", f.stat().st_size)
                  for f in listing if f.is_file() and not f.name.startswith("."))


def _shard(root: Path, name: str, scenes: list[str]) -> Shard:
    files = [f for s in scenes for f in _scene_files(root, s)]
    key = hashlib.sha256("".join(f"{p}\t{n}\n" for p, n in files).encode()).hexdigest()
    return Shard(name=name, scenes=scenes, files=len(files), bytes=sum(n for _, n in files), content=key,
                 paths=[p for p, _ in files])


def plan_shards(root: str | Path, scenes_per_shard: int) -> list[Shard]:
    """The scenes of the cache in name order, cut into runs of `scenes_per_shard`."""
    entries = Path(root) / "entries"
    scenes = sorted(d.name for d in os.scandir(entries) if d.is_dir()) if entries.is_dir() else []
    if not scenes:
        raise ValueError(f"{root} holds no entries")
    return [_shard(Path(root), f"{i // scenes_per_shard:05d}", scenes[i:i + scenes_per_shard])
            for i in range(0, len(scenes), scenes_per_shard)]


class _HashingWriter:
    def __init__(self, f):
        self.f, self.hash, self.bytes = f, hashlib.sha256(), 0

    def write(self, data) -> int:
        self.hash.update(data)
        self.bytes += len(data)
        return self.f.write(data)

    def flush(self) -> None:
        self.f.flush()


def _cpus() -> int:
    """The CPUs this process may use: a Slurm job's share, not the whole node."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def write_shard(root: str | Path, shard: Shard, path: str | Path, level: int = 3) -> None:
    """Write the files of a shard as a zstd tar at `path`, and fill in its archive size and SHA-256."""
    import zstandard

    root, path = Path(root), Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "wb") as f:
            out = _HashingWriter(f)
            with zstandard.ZstdCompressor(level=level, threads=_cpus()).stream_writer(out, closefd=False) as z:
                with tarfile.open(fileobj=z, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                    for rel in shard.paths:
                        tar.add(root / rel, arcname=rel, recursive=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    shard.archive_bytes, shard.sha256 = out.bytes, out.hash.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(READ_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def extract_shard(path: str | Path, root: str | Path) -> None:
    """Extract a shard into a cache folder."""
    import zstandard

    with open(path, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as z:
        with tarfile.open(fileobj=z, mode="r|") as tar:
            tar.extractall(root, filter="data")


# ----- upload -----------------------------------------------------------------


@dataclass
class UploadReport:
    repo_id: str
    cache: str
    shards: int = 0
    uploaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    bytes: int = 0  # of the cache files in all shards
    archive_bytes: int = 0  # of all shards in the repository
    seconds: float = 0.0


def _manifest(name: str, fingerprint: str, cfg: BackupConfig, shards: list[dict], complete: bool) -> bytes:
    return (json.dumps({
        "manifest_version": MANIFEST_VERSION,
        "cache": name,
        "fingerprint": fingerprint,
        "scenes_per_shard": cfg.scenes_per_shard,
        "compression": f"tar, zstd level {cfg.level}",
        "complete": complete,
        "shards": shards,
    }, indent=1) + "\n").encode()


def remote_manifest(api, repo_id: str, name: str) -> dict[str, Any] | None:
    """The manifest of cache `name` in the repository, or None when it has none."""
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = api.hf_hub_download(repo_id, f"{name}/manifest.json", repo_type=REPO_TYPE)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    return json.loads(Path(path).read_text())


DATASET_CARD = """\
---
license: cc-by-nc-sa-4.0
pretty_name: confhead cache
tags:
- depth-estimation
- nuscenes
---

# confhead cache

Frozen Depth Anything V2 outputs on nuScenes, cached for the training of a
per-pixel confidence head: decoder features, disparity, temporal spread, and
the lidar and radar points of each camera image. Each folder is one cache,
built by [confhead](https://github.com/kgqiao/Vision-Project-CSCI-566/tree/main/confhead).

The cache derives from [nuScenes](https://www.nuscenes.org/), which is under
CC BY-NC-SA 4.0 and the nuScenes terms of use. The same terms apply here.

## Restore a cache

    confhead cache download <cache> <folder>

Or by hand: download `<cache>/` and extract each shard into one folder.

    hf download {repo_id} --repo-type dataset --include "<cache>/*" --local-dir .
    for f in <cache>/shards/*.tar.zst; do tar --zstd -xf "$f" -C <cache>; done
"""


def upload(root: str | Path, cfg: BackupConfig, api=None) -> UploadReport:
    """Upload a cache to the repository of `cfg`: the shards that changed, then the index and the header.

    The token comes from `hf auth login` or $HF_TOKEN. The index must be current,
    so the shards and the index describe the same entries.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    t0 = time.perf_counter()
    repo_id = _require_repo(cfg)
    root = Path(root).resolve()
    name = root.name
    header = json.loads((root / "cache.json").read_text())
    index = root / "index.json"
    if not index.is_file() or json.loads(index.read_text())["fingerprint"] != header["fingerprint"]:
        raise ValueError(f"{root}/index.json is missing or out of date; "
                         "write it with examples/build_cache.py --check first")

    api = api or HfApi()
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=cfg.private, exist_ok=True)
    in_repo = set(api.list_repo_files(repo_id, repo_type=REPO_TYPE))
    remote = remote_manifest(api, repo_id, name) if f"{name}/manifest.json" in in_repo else None
    # What the repository holds now, by shard name; each upload replaces one record.
    held = {s["name"]: s for s in (remote or {}).get("shards", [])
            if f"{name}/shards/{s['name']}.tar.zst" in in_repo}

    shards = plan_shards(root, cfg.scenes_per_shard)
    staging = Path(cfg.staging) if cfg.staging else root.with_name(f"{name}.staging")
    report = UploadReport(repo_id=repo_id, cache=name, shards=len(shards))
    log.info("%s: %d shards of %d scenes, %s", name, len(shards), cfg.scenes_per_shard,
             f"{len(held)} already in {repo_id}" if held else f"none in {repo_id} yet")

    def header_ops() -> list:  # an operation is spent once it is committed
        return [CommitOperationAdd(f"{name}/{f}", root / f) for f in HEADER_FILES]

    for i, shard in enumerate(shards, 1):
        old = held.get(shard.name)
        if old and old["content"] == shard.content:
            shard.archive_bytes, shard.sha256 = old["archive_bytes"], old["sha256"]
            report.skipped.append(shard.name)
            continue
        path = staging / f"{shard.name}.tar.zst"
        t = time.perf_counter()
        write_shard(root, shard, path, cfg.level)
        packed = time.perf_counter() - t
        held[shard.name] = shard.record()
        manifest = _manifest(name, header["fingerprint"], cfg, sorted(held.values(), key=lambda s: s["name"]),
                             complete=False)
        t = time.perf_counter()
        try:
            api.create_commit(repo_id, repo_type=REPO_TYPE, commit_message=f"{name}: shard {shard.name}", operations=[
                CommitOperationAdd(f"{name}/shards/{shard.name}.tar.zst", path),
                CommitOperationAdd(f"{name}/manifest.json", manifest),
                *header_ops(),
            ])
        finally:
            path.unlink(missing_ok=True)
        report.uploaded.append(shard.name)
        log.info("shard %s (%d/%d): %d files, %s to %s (%.0f%%), packed in %.0f s, sent in %.0f s",
                 shard.name, i, len(shards), shard.files, _gb(shard.bytes), _gb(shard.archive_bytes),
                 100 * shard.archive_bytes / max(shard.bytes, 1), packed, time.perf_counter() - t)

    planned = {s.name for s in shards}
    report.removed = sorted(set(held) - planned)
    ops = [*header_ops(), CommitOperationAdd(f"{name}/index.json", index)]
    ops.append(CommitOperationAdd(f"{name}/manifest.json",
                                  _manifest(name, header["fingerprint"], cfg, [s.record() for s in shards], True)))
    ops += [CommitOperationDelete(f"{name}/shards/{s}.tar.zst") for s in report.removed]
    if "README.md" not in in_repo:
        ops.append(CommitOperationAdd("README.md", DATASET_CARD.replace("{repo_id}", repo_id).encode()))
    api.create_commit(repo_id, repo_type=REPO_TYPE, operations=ops, commit_message=f"{name}: index and manifest")
    shutil.rmtree(staging, ignore_errors=True)

    report.bytes = sum(s.bytes for s in shards)
    report.archive_bytes = sum(s.archive_bytes for s in shards)
    report.seconds = time.perf_counter() - t0
    return report


def _gb(n: float) -> str:
    return f"{n / 1e9:,.2f} GB"


def format_upload(r: UploadReport) -> str:
    removed = f", {len(r.removed)} removed" if r.removed else ""
    lines = [
        f"repository   https://huggingface.co/datasets/{r.repo_id}/tree/main/{r.cache}",
        f"shards       {r.shards}: {len(r.uploaded)} uploaded, {len(r.skipped)} unchanged{removed}",
        f"size         {_gb(r.bytes)} of cache files in {_gb(r.archive_bytes)} of shards "
        f"({100 * r.archive_bytes / max(r.bytes, 1):.0f}%)",
        f"time         {r.seconds / 60:.1f} min",
    ]
    return "\n".join(lines)


# ----- download ---------------------------------------------------------------


@dataclass
class DownloadReport:
    repo_id: str
    cache: str
    root: Path
    complete: bool = True
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _have(root: Path, record: dict[str, Any]) -> bool:
    return _shard(root, record["name"], record["scenes"]).content == record["content"]


def download(cfg: BackupConfig, name: str, root: str | Path, api=None) -> DownloadReport:
    """Restore cache `name` from the repository of `cfg` into the folder `root`.

    A shard whose files are all in `root` already is not downloaded again, so a
    stopped download continues. From an upload that did not finish, it restores
    the shards that are up and writes the index of their entries. Each shard is checked against the SHA-256 of the
    manifest before it is extracted, and deleted after.
    """
    from huggingface_hub import HfApi

    t0 = time.perf_counter()
    repo_id = _require_repo(cfg)
    api = api or HfApi()
    root = Path(root).resolve()
    manifest = remote_manifest(api, repo_id, name)
    if manifest is None:
        raise FileNotFoundError(f"{repo_id} holds no cache {name!r}")
    staging = Path(cfg.staging) if cfg.staging else root.with_name(f"{name}.staging")
    report = DownloadReport(repo_id=repo_id, cache=name, root=root, complete=manifest["complete"])
    if not manifest["complete"]:
        log.warning("%s: the upload of %s did not finish; restoring the shards it holds", repo_id, name)

    shards = manifest["shards"]
    for i, record in enumerate(shards, 1):
        if _have(root, record):
            report.skipped.append(record["name"])
            continue
        t = time.perf_counter()
        path = Path(api.hf_hub_download(repo_id, f"{name}/shards/{record['name']}.tar.zst",
                                        repo_type=REPO_TYPE, local_dir=staging))
        try:
            if _sha256(path) != record["sha256"]:
                raise OSError(f"{path}: SHA-256 differs from the manifest; download it again")
            extract_shard(path, root)
        finally:
            path.unlink(missing_ok=True)
        report.downloaded.append(record["name"])
        log.info("shard %s (%d/%d): %s, restored in %.0f s",
                 record["name"], i, len(shards), _gb(record["bytes"]), time.perf_counter() - t)

    for f in HEADER_FILES + (("index.json",) if report.complete else ()):
        src = Path(api.hf_hub_download(repo_id, f"{name}/{f}", repo_type=REPO_TYPE, local_dir=staging))
        shutil.move(src, root / f)
    if not report.complete:
        build_index(root)
    shutil.rmtree(staging, ignore_errors=True)
    report.seconds = time.perf_counter() - t0
    return report


def format_download(r: DownloadReport) -> str:
    lines = [
        f"cache        {r.root}",
        f"shards       {len(r.downloaded)} downloaded, {len(r.skipped)} already on disk",
        f"time         {r.seconds / 60:.1f} min",
    ]
    if not r.complete:
        lines.append("incomplete   the upload of this cache did not finish, so this restores only part of it; "
                     "finish the upload, then download again")
    return "\n".join(lines)

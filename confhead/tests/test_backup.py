import json
import shutil
import subprocess
from pathlib import Path

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import EntryNotFoundError

from confhead.cache.backup import BackupConfig, download, load_backup_config, plan_shards, upload, write_shard
from confhead.cache.config import CacheConfig
from confhead.cache.report import format_info, info
from confhead.cache.store import Cache, build_index, open_for_build, write_entry
from test_cache import make_entry

SCENES = ["scene-a", "scene-b", "scene-c"]


class FakeHub:
    """The calls of HfApi that the backup makes, on a folder that stands for the repository."""

    def __init__(self, folder: Path):
        self.repo = folder / "repo"
        self.downloads = folder / "downloads"
        self.commits: list[str] = []

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.repo.mkdir(parents=True, exist_ok=True)

    def list_repo_files(self, repo_id, repo_type):
        return sorted(str(p.relative_to(self.repo)) for p in self.repo.rglob("*") if p.is_file())

    def hf_hub_download(self, repo_id, filename, repo_type, local_dir=None):
        src = self.repo / filename
        if not src.is_file():
            raise EntryNotFoundError(filename)
        dst = Path(local_dir or self.downloads) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        return str(dst)

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                assert not op._is_committed, f"{op.path_in_repo}: an operation can be committed once only"
                op._is_committed = True
            target = self.repo / op.path_in_repo
            if isinstance(op, CommitOperationDelete):
                target.unlink()
            elif isinstance(op, CommitOperationAdd):
                target.parent.mkdir(parents=True, exist_ok=True)
                data = op.path_or_fileobj
                target.write_bytes(data if isinstance(data, bytes) else Path(data).read_bytes())
        self.commits.append(commit_message)


def make_cache(root: Path, scenes=SCENES, per_scene=2) -> Path:
    cfg = CacheConfig()
    cfg.output.root = str(root)
    cfg.splits = {"train": scenes[:-1], "val": scenes[-1:]}
    open_for_build(cfg, overwrite=False)
    for scene in scenes:
        for i in range(per_scene):
            add_entry(root, cfg, scene, i)
    build_index(root)
    return root


def add_entry(root: Path, cfg: CacheConfig, scene: str, i: int) -> None:
    arrays, meta = make_entry(scene=scene, key=f"{scene}_{i:03d}_CAM_FRONT")
    meta["fingerprint"] = cfg.fingerprint()
    write_entry(root, scene, meta["key"], arrays, meta)


def files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def cache(tmp_path):
    return make_cache(tmp_path / "nuscenes-test")


def test_info_counts_entries_and_bytes(cache):
    (cache / "entries/scene-a/.scene-a_009_CAM_FRONT.npz.abc.tmp").write_bytes(b"x" * 10)
    v = info(cache)
    assert (v["scenes"], v["pairs"], v["npz"], v["json"], v["temp_files"]) == (3, 6, 6, 6, 1)
    assert v["bytes_total"] == sum(len(b) for b in files(cache).values())
    assert v["index"] == {"entries": 6, "current": True, "stale_entries": 0, "tags": {"train": 4, "val": 2}}
    assert "6 complete entries in 3 scenes" in format_info(v)


def test_upload_then_download_restores_every_file(cache, tmp_path):
    hub = FakeHub(tmp_path)
    cfg = BackupConfig(repo_id="team/confhead-cache", scenes_per_shard=2, staging=str(tmp_path / "staging"))
    report = upload(cache, cfg, api=hub)
    assert report.uploaded == ["00000", "00001"] and not report.skipped
    manifest = json.loads((hub.repo / "nuscenes-test/manifest.json").read_text())
    assert manifest["complete"] and [s["scenes"] for s in manifest["shards"]] == [SCENES[:2], SCENES[2:]]
    assert {"README.md", "nuscenes-test/index.json", "nuscenes-test/shards/00001.tar.zst"} <= set(hub.list_repo_files("", ""))
    assert not (tmp_path / "staging").exists()

    restored = tmp_path / "restored"
    got = download(cfg, "nuscenes-test", restored, api=hub)
    assert got.downloaded == ["00000", "00001"] and got.complete
    assert files(restored) == files(cache)
    assert len(Cache(restored)) == 6
    assert download(cfg, "nuscenes-test", restored, api=hub).skipped == ["00000", "00001"]


def test_upload_sends_only_the_shards_that_changed(cache, tmp_path):
    hub = FakeHub(tmp_path)
    cfg = BackupConfig(repo_id="team/confhead-cache", scenes_per_shard=2)
    upload(cache, cfg, api=hub)
    assert upload(cache, cfg, api=hub).skipped == ["00000", "00001"]

    add_entry(cache, CacheConfig(), "scene-c", 5)
    build_index(cache)
    again = upload(cache, cfg, api=hub)
    assert again.uploaded == ["00001"] and again.skipped == ["00000"]

    cfg.scenes_per_shard = 3
    fewer = upload(cache, cfg, api=hub)
    assert fewer.uploaded == ["00000"] and fewer.removed == ["00001"]
    assert not (hub.repo / "nuscenes-test/shards/00001.tar.zst").exists()


def test_a_stopped_upload_continues_and_a_download_says_it_is_incomplete(cache, tmp_path, monkeypatch):
    hub = FakeHub(tmp_path)
    cfg = BackupConfig(repo_id="team/confhead-cache", scenes_per_shard=1)
    commit = hub.create_commit

    def fail_on_second(*args, **kwargs):
        if len(hub.commits) == 1:
            raise OSError("connection reset")
        commit(*args, **kwargs)

    monkeypatch.setattr(hub, "create_commit", fail_on_second)
    with pytest.raises(OSError):
        upload(cache, cfg, api=hub)
    assert not list((cache.parent / "nuscenes-test.staging").glob("*.tar.zst"))
    manifest = json.loads((hub.repo / "nuscenes-test/manifest.json").read_text())
    assert not manifest["complete"] and [s["name"] for s in manifest["shards"]] == ["00000"]

    partial = download(cfg, "nuscenes-test", tmp_path / "partial", api=hub)
    assert not partial.complete and partial.downloaded == ["00000"]
    assert [r["scene"] for r in Cache(tmp_path / "partial").rows()] == ["scene-a", "scene-a"]

    monkeypatch.setattr(hub, "create_commit", commit)
    report = upload(cache, cfg, api=hub)
    assert report.skipped == ["00000"] and report.uploaded == ["00001", "00002"]


def test_upload_refuses_an_index_that_is_out_of_date(cache, tmp_path):
    (cache / "index.json").unlink()
    with pytest.raises(ValueError, match="index.json is missing or out of date"):
        upload(cache, BackupConfig(repo_id="team/x"), api=FakeHub(tmp_path))


def test_download_checks_the_sha256(cache, tmp_path):
    hub = FakeHub(tmp_path)
    cfg = BackupConfig(repo_id="team/confhead-cache")
    upload(cache, cfg, api=hub)
    (hub.repo / "nuscenes-test/shards/00000.tar.zst").write_bytes(b"not the shard")
    with pytest.raises(OSError, match="SHA-256 differs"):
        download(cfg, "nuscenes-test", tmp_path / "restored", api=hub)


@pytest.mark.skipif(subprocess.run(["tar", "--zstd", "--version"], capture_output=True).returncode != 0
                    or not shutil.which("zstd"), reason="tar with zstd is not installed")
def test_a_shard_extracts_with_plain_tar(cache, tmp_path):
    shard = plan_shards(cache, 10)[0]
    write_shard(cache, shard, tmp_path / "s.tar.zst")
    out = tmp_path / "out"
    out.mkdir()
    subprocess.run(["tar", "--zstd", "-xf", tmp_path / "s.tar.zst", "-C", out], check=True)
    assert files(out) == {k: v for k, v in files(cache).items() if k.startswith("entries/")}


def test_backup_config_names_the_bad_setting(tmp_path):
    path = tmp_path / "hf.toml"
    path.write_text('[huggingface]\nrepo_id = "team/x"\n[shards]\nscenes_per_shard = 10\nstaging = "stage"\n')
    cfg = load_backup_config(path)
    assert (cfg.repo_id, cfg.scenes_per_shard, cfg.staging) == ("team/x", 10, str(tmp_path / "stage"))
    path.write_text("[shards]\nscene_per_shard = 10\n")
    with pytest.raises(ValueError, match="unknown setting shards.scene_per_shard"):
        load_backup_config(path)
    path.write_text('[shards]\nlevel = "high"\n')
    with pytest.raises(ValueError, match="shards.level must be an integer"):
        load_backup_config(path)
    with pytest.raises(ValueError, match="Set huggingface.repo_id"):
        upload(tmp_path, BackupConfig(), api=None)


def test_the_shipped_backup_config_loads():
    cfg = load_backup_config(Path(__file__).parents[1] / "configs/huggingface.toml")
    assert cfg.repo_id == "" and not cfg.private

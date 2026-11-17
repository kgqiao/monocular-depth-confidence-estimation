import json
from types import SimpleNamespace

import pytest

from confhead import matrix as mx
from confhead.cache.store import Cache
from confhead.hub import RunRecord
from confhead.train import Training
from test_train import make_cache

MANIFEST = """\
config = "train.toml"
repo = "team/confhead"
time = "02:00:00"
cache_entries = 12

[[run]]
name = "base"
stage = 0
group = "baseline"
set = []

[[run]]
name = "m1-no-radar"
stage = 1
group = "inputs"
set = ["inputs.radar=false"]
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A manifest of two runs over a cache of 12 entries, in a project folder with no Slurm and no network."""
    cache = make_cache(tmp_path / "cache")
    (tmp_path / "train.toml").write_text(
        f"[data]\ncache = '{cache}'\n[head]\nwidths = [4]\n[train]\nepochs = 2\nbatch_size = 2\ndev_fraction = 0.25\n"
        f"[output]\nruns = '{tmp_path / 'runs'}'\n[run]\ndevice = 'cpu'\nnum_workers = 0\n")
    (tmp_path / "matrix.toml").write_text(MANIFEST)
    monkeypatch.setattr(mx, "PROJECT", tmp_path)
    monkeypatch.setattr(mx, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(mx, "check_write_access", lambda repo: "me")
    monkeypatch.setenv("SBATCH_ACCOUNT", "project")
    return mx.load_matrix(tmp_path / "matrix.toml")


def plans(m, records=None, jobs=None, trial=False):
    return mx.make_plans(m, m.runs, mx.open_cache(m), records or {}, jobs or {}, trial)


def train(m, name, epochs=None, trial=False):
    cfg = mx.run_config(m, mx.select(m, patterns=[name])[0], trial)
    full = cfg.train.epochs
    if epochs:
        cfg.train.epochs = epochs
    training = Training(cfg)
    training.fit()
    if epochs:  # a run that stopped: its config names all the epochs
        path = training.dir / "config.json"
        config = json.loads(path.read_text())
        config["settings"]["train"]["epochs"] = full
        path.write_text(json.dumps(config))
    return training


def test_the_manifest_names_its_mistakes(project, tmp_path):
    assert [r.name for r in mx.select(project, stages=["1"])] == ["m1-no-radar"]
    assert [r.name for r in mx.select(project, patterns=["m1-*", "base"])] == ["base", "m1-no-radar"]
    with pytest.raises(ValueError, match="no run of matrix.toml matches"):
        mx.select(project, patterns=["m9-*"])
    path = tmp_path / "bad.toml"
    path.write_text(MANIFEST + MANIFEST.split("\n\n", 1)[1])
    with pytest.raises(ValueError, match="run names used twice: base, m1-no-radar"):
        mx.load_matrix(path)
    path.write_text(MANIFEST.replace('name = "base"', 'name = "trial-base"'))
    with pytest.raises(ValueError, match="must not be empty"):
        mx.load_matrix(path)
    path.write_text(MANIFEST.replace("cache_entries = 12", "cache_entries = 16"))
    with pytest.raises(ValueError, match="holds 12 entries, and bad.toml expects 16"):
        mx.open_cache(mx.load_matrix(path))


def test_a_trial_is_one_short_epoch_under_its_own_name(project):
    cfg = mx.run_config(project, project.runs[1], trial=True)
    assert (cfg.output.name, cfg.train.epochs, cfg.train.samples_per_epoch) == ("trial-m1-no-radar", 1, mx.TRIAL_IMAGES)
    assert not cfg.inputs.radar
    full = mx.run_config(project, project.runs[1])
    assert (full.output.name, full.output.group, full.train.epochs) == ("m1-no-radar", "inputs", 2)


def test_the_state_of_a_run_comes_from_its_folder_the_repository_and_the_queue(project):
    base, other = plans(project)
    assert base.action == other.action == "train" and base.fingerprint != other.fingerprint

    train(project, "base", epochs=1)
    base = plans(project)[0]
    assert (base.action, base.epochs_done) == ("resume", 1)
    assert plans(project, jobs={"base": "77"})[0].action == "active"

    train(project, "base")
    base = plans(project)[0]
    assert base.action == "upload" and base.describe() == "upload only"
    record = RunRecord.read(base.dir)
    assert plans(project, {"base": record})[0].action == "done"

    record.config["train_images"] += 100  # the same name, trained on another cache
    base = plans(project, {"base": record})[0]
    assert (base.action, base.remote) == ("upload", "other") and "replaces the repository run" in base.describe()

    config = base.dir / "config.json"
    config.write_text(json.dumps({**json.loads(config.read_text()), "train_images": 1}))
    base = plans(project)[0]
    assert (base.action, base.local) == ("train", "stale")


def test_a_finished_run_in_the_repository_needs_no_folder_here(project):
    training = train(project, "base")
    record = RunRecord.read(training.dir)
    (training.dir / "last.pt").unlink()
    base = plans(project, {"base": record})[0]
    assert (base.local, base.action, base.best) == ("none", "done", record.best["dev_loss"])


class Slurm:
    """Records the sbatch calls and numbers the jobs from 101."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if args[0] == "squeue":
            return ""
        return f"{100 + sum(c[0] == 'sbatch' for c in self.calls)}\n"


@pytest.fixture
def slurm(monkeypatch):
    fake = Slurm()
    monkeypatch.setattr(mx, "_slurm", fake)
    monkeypatch.setattr(mx, "_sync", lambda: None)
    return fake


def test_run_submits_a_job_per_run_and_a_report_after_them(project, slurm, monkeypatch, capsys):
    monkeypatch.setattr(mx, "download_runs", lambda repo: {})
    mx.submit(project, project.runs)
    jobs = [c for c in slurm.calls if c[0] == "sbatch"]
    assert jobs[0] == ("sbatch", "--parsable", "--job-name=confhead-base", "--time=02:00:00",
                       "--output=logs/train/base-%j.out", mx.TRAIN_JOB, str(project.path), "base", "--repo", "team/confhead")
    assert jobs[1][2] == "--job-name=confhead-m1-no-radar"
    assert jobs[2][:3] == ("sbatch", "--parsable", "--dependency=afterany:101:102")
    assert jobs[2][-5:] == ("--jobs", "101,102", "--runs", "base", "m1-no-radar")
    assert "report    job 103" in capsys.readouterr().out


def test_run_uploads_a_finished_run_and_trains_only_the_rest(project, slurm, monkeypatch):
    training = train(project, "base")
    uploads = []
    monkeypatch.setattr(mx, "download_runs", lambda repo: {})
    monkeypatch.setattr(mx, "upload_run", lambda run, repo, replace: uploads.append(run) or SimpleNamespace(commit_url="u"))
    mx.submit(project, project.runs)
    assert uploads == [training.dir]
    assert [c[2] for c in slurm.calls if c[0] == "sbatch"][:1] == ["--job-name=confhead-m1-no-radar"]


def test_run_replaces_and_archives_only_when_told(project, slurm, monkeypatch, capsys):
    training = train(project, "base")
    record = RunRecord.read(training.dir)
    record.config["train_images"] += 100
    config = training.dir / "config.json"
    config.write_text(json.dumps({**json.loads(config.read_text()), "train_images": 1}))
    monkeypatch.setattr(mx, "download_runs", lambda repo: {"base": record})
    with pytest.raises(ValueError) as e:
        mx.submit(project, project.runs)
    assert "--archive-stale renames the folder" in str(e.value) and "--replace overwrites it" in str(e.value)
    assert not [c for c in slurm.calls if c[0] == "sbatch"] and training.dir.is_dir()
    assert not mx.preview(project, project.runs)

    tags = []
    monkeypatch.setattr(mx, "tag_state", lambda repo, tag: tags.append(tag) or tag)
    mx.submit(project, project.runs, replace=True, archive_stale=True, tag="cache-149k")
    assert tags == ["cache-149k"] and not training.dir.exists()
    assert [p.name for p in training.dir.parent.iterdir()][0].startswith("base.stale-")
    assert slurm.calls[-3][-1] == "--replace"


def test_run_refuses_changed_code_and_a_missing_account(project, slurm, monkeypatch):
    monkeypatch.setattr(mx, "download_runs", lambda repo: {})
    monkeypatch.setattr(mx, "git_commit", lambda: "abc1234-dirty")
    monkeypatch.delenv("SBATCH_ACCOUNT")
    with pytest.raises(ValueError, match="(?s)SBATCH_ACCOUNT is not set.*tracked files have changes"):
        mx.submit(project, project.runs)


def test_the_worker_trains_and_uploads_each_epoch(project, monkeypatch, capsys):
    uploads = []
    monkeypatch.setattr(mx, "download_runs", lambda repo: {})
    monkeypatch.setattr(mx, "upload_run", lambda run, repo, partial, replace: uploads.append(partial)
                        or SimpleNamespace(commit_url="u"))
    mx.worker(project, project.runs[0])
    assert uploads == [True, False, False]  # after each epoch, and once more at the end
    assert "best      epoch" in capsys.readouterr().out

    def fail(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(mx, "upload_run", fail)
    monkeypatch.setattr(mx.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="base is trained but not in team/confhead"):
        mx.worker(project, project.runs[0])


def test_the_worker_stops_before_training_over_another_model(project, monkeypatch):
    record = RunRecord.read(train(project, "base", trial=True).dir)
    record.config["train_images"] += 100
    monkeypatch.setattr(mx, "download_runs", lambda repo: {"trial-base": record})
    with pytest.raises(ValueError, match="holds another model under trial-base"):
        mx.worker(project, project.runs[0], trial=True)


def test_the_report_names_what_failed_with_the_end_of_its_log(project, monkeypatch, capsys):
    done = train(project, "base")
    train(project, "m1-no-radar", epochs=1)
    monkeypatch.setattr(mx, "download_runs", lambda repo: {"base": RunRecord.read(done.dir)})
    monkeypatch.setattr(mx, "active_jobs", lambda: {})
    monkeypatch.setattr(mx, "_accounting", lambda ids: {"101": ("COMPLETED", "01:00:00", "0:0"),
                                                         "102": ("TIMEOUT", "02:00:10", "0:0")})
    logs = project.path.parent / mx.LOGS
    logs.mkdir(parents=True)
    (logs / "m1-no-radar-102.out").write_text("epoch 1: train loss\nslurmstepd: CANCELLED DUE TO TIME LIMIT\n")
    assert not mx.status(project, project.runs, job_ids=["101", "102"])
    out = capsys.readouterr().out
    assert "1 of 2 runs are done" in out and "## m1-no-radar: resume after epoch 1 of 2" in out
    assert "Job 102: TIMEOUT" in out and "    slurmstepd: CANCELLED DUE TO TIME LIMIT" in out
    assert "## base" not in out


def test_the_estimate_scales_the_earlier_run_of_the_name(project):
    training = train(project, "base")
    record = RunRecord.read(training.dir)
    for h in record.history:
        h["minutes"] = 30.0
    record.config["train_images"] //= 2  # the earlier run saw half the images in an hour
    plan = plans(project)[1]
    by_prior, slow = mx.estimate_hours(plan, {"m1-no-radar": record})
    assert by_prior == pytest.approx(2.0) and slow == pytest.approx(2 * plan.train_images / mx.SLOW_RATE / 3600)
    assert mx.estimate_hours(plan, {})[0] is None

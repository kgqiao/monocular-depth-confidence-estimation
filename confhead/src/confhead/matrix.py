"""The run matrix: a manifest of training runs, the state of each, and their Slurm jobs.

    confhead train preview configs/run-matrix.toml
    confhead train run     configs/run-matrix.toml
    confhead train status  configs/run-matrix.toml

The state of a run is read each time from three places: its run folder, the
Hugging Face repository, and the Slurm queue. No file records what was
submitted, so the same command sent again does only what is left.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache.store import Cache
from .hub import RunRecord, check_write_access, delete_runs, download_runs, tag_state, upload_run
from .train import TrainConfig, Training, config_fingerprint, git_commit, load_train_config, planned_fingerprint, validate

log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[2]
TRAIN_JOB = "examples/carc-train.sbatch"
REPORT_JOB = "examples/carc-train-report.sbatch"
LOGS = "logs/train"
JOB_PREFIX = "confhead-"
TRIAL_PREFIX = "trial-"
TRIAL_IMAGES = 10_000
# The slowest loading measured on CARC, in images per second (Docs/worklog.md): the worst case of the estimate.
SLOW_RATE = 28
FAILED_STATES = ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE")


@dataclass
class Run:
    name: str
    stage: str = ""
    group: str = ""
    question: str = ""
    set: list[str] = field(default_factory=list)


@dataclass
class Matrix:
    path: Path
    config: Path
    repo: str
    time: str
    cache_entries: int  # the entries a complete cache holds; 0: not checked
    runs: list[Run]


def load_matrix(path: str | Path) -> Matrix:
    path = Path(path).expanduser().resolve()
    data = tomllib.loads(path.read_text())
    known = {"config", "repo", "time", "cache_entries", "run"}
    if set(data) - known:
        raise ValueError(f"{path}: unknown keys {sorted(set(data) - known)}; known: {', '.join(sorted(known))}")
    for key in ("config", "repo"):
        if not data.get(key):
            raise ValueError(f"{path}: {key} is missing")
    runs = []
    for row in data.get("run", []):
        try:
            run = Run(**{**row, "stage": str(row.get("stage", ""))})
        except TypeError:
            raise ValueError(f"{path}: a [[run]] takes name, stage, group, question, set; got {sorted(row)}") from None
        if not run.name or run.name.startswith(TRIAL_PREFIX) or "/" in run.name:
            raise ValueError(f"{path}: run name {run.name!r} must not be empty, hold a slash, or start with {TRIAL_PREFIX}")
        runs.append(run)
    names = [r.name for r in runs]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"{path}: run names used twice: {', '.join(twice)}")
    if not runs:
        raise ValueError(f"{path} lists no [[run]]")
    return Matrix(path, (path.parent / data["config"]).resolve(), data["repo"], data.get("time", "16:00:00"),
                  int(data.get("cache_entries", 0)), runs)


def select(matrix: Matrix, stages: list[str] | None = None, patterns: list[str] | None = None) -> list[Run]:
    """The runs of the given stages and name patterns; every run when neither is given."""
    runs = [r for r in matrix.runs
            if (not stages or r.stage in stages) and (not patterns or any(fnmatch.fnmatch(r.name, p) for p in patterns))]
    if not runs:
        raise ValueError(f"no run of {matrix.path.name} matches; it holds {', '.join(r.name for r in matrix.runs)}")
    return runs


def run_config(matrix: Matrix, run: Run, trial: bool = False) -> TrainConfig:
    """The settings of a run. A trial is one short epoch under the name trial-<run>."""
    cfg = load_train_config(matrix.config, run.set)
    cfg.output.name, cfg.output.group = run.name, run.group
    if trial:
        cfg.output.name, cfg.output.group = TRIAL_PREFIX + run.name, "trial"
        cfg.train.epochs = 1
        cfg.train.samples_per_epoch = min(cfg.train.samples_per_epoch or TRIAL_IMAGES, TRIAL_IMAGES)
    validate(cfg)
    return cfg


@dataclass
class Plan:
    run: Run
    cfg: TrainConfig
    fingerprint: str
    train_images: int
    epochs_done: int = 0
    local: str = "none"  # none, partial, complete, stale
    remote: str = "none"  # none, partial, complete, other
    job: str = ""  # the Slurm job of this run that is queued or running
    best: float | None = None

    @property
    def name(self) -> str:
        return self.cfg.output.name

    @property
    def dir(self) -> Path:
        return Path(self.cfg.output.runs) / self.name

    @property
    def images_per_epoch(self) -> int:
        return self.cfg.train.samples_per_epoch or self.train_images

    @property
    def action(self) -> str:
        """train, resume, upload, done, or active."""
        if self.job:
            return "active"
        if self.remote == "complete":
            return "done"
        if self.local == "complete":
            return "upload"
        return "resume" if self.local == "partial" else "train"

    def describe(self) -> str:
        epochs = self.cfg.train.epochs
        text = {"train": "train", "resume": f"resume after epoch {self.epochs_done} of {epochs}",
                "upload": "upload only", "done": "done, in the repository", "active": f"job {self.job}"}[self.action]
        if self.action in ("train", "resume", "upload"):
            if self.local == "stale":
                text += "; stale run folder"
            if self.remote == "other":
                text += "; replaces the repository run"
        return text


def _local(plan: Plan) -> None:
    config, history = plan.dir / "config.json", plan.dir / "history.json"
    if not (plan.dir / "last.pt").is_file() or not config.is_file():
        return
    if config_fingerprint(json.loads(config.read_text())) != plan.fingerprint:
        plan.local = "stale"
        return
    done = json.loads(history.read_text()) if history.is_file() else []
    plan.epochs_done = len(done)
    plan.local = "complete" if len(done) >= plan.cfg.train.epochs else "partial"
    plan.best = min((h["dev_loss"] for h in done), default=None)


def _remote(plan: Plan, record: RunRecord | None) -> None:
    if record is None:
        return
    if record.fingerprint != plan.fingerprint:
        plan.remote = "other"
        return
    plan.remote = "complete" if record.complete else "partial"
    if plan.best is None and record.history:
        plan.best = record.best["dev_loss"]


def make_plans(matrix: Matrix, runs: list[Run], cache: Cache, records: dict[str, RunRecord], jobs: dict[str, str],
               trial: bool = False) -> list[Plan]:
    plans = []
    for run in runs:
        cfg = run_config(matrix, run, trial)
        fingerprint, train_images = planned_fingerprint(cfg, cache)
        plan = Plan(run, cfg, fingerprint, train_images, job=jobs.get(cfg.output.name, ""))
        _local(plan)
        _remote(plan, records.get(plan.name))
        plans.append(plan)
    return plans


def estimate_hours(plan: Plan, records: dict[str, RunRecord]) -> tuple[float | None, float]:
    """Hours left: by the speed of the earlier run of this name, if the repository has one, and at the slowest rate."""
    images = plan.images_per_epoch * (plan.cfg.train.epochs - plan.epochs_done)
    if plan.action not in ("train", "resume"):
        return None, 0.0
    prior = records.get(plan.run.name)
    by_prior = None
    if prior and prior.history:
        train = prior.config["settings"]["train"]
        seen = (train["samples_per_epoch"] or prior.config["train_images"]) * len(prior.history)
        by_prior = sum(h["minutes"] for h in prior.history) / 60 / seen * images
    return by_prior, images / SLOW_RATE / 3600


def _slurm(*args: str) -> str:
    return subprocess.run(args, cwd=PROJECT, capture_output=True, text=True, check=True).stdout


def _sync() -> None:
    """Fill the uv cache that each job installs its environment from."""
    subprocess.run(["uv", "sync", "--quiet"], cwd=PROJECT, check=True)


def active_jobs() -> dict[str, str] | None:
    """Run name to job number, for the queued and running jobs of this user; None where there is no Slurm."""
    try:
        out = _slurm("squeue", "--me", "--noheader", "--format=%i|%j")
    except FileNotFoundError:
        return None
    jobs = {}
    for line in out.splitlines():
        job, _, name = line.strip().partition("|")
        if name.startswith(JOB_PREFIX):
            jobs[name[len(JOB_PREFIX):]] = job
    return jobs


def open_cache(matrix: Matrix) -> Cache:
    cache = Cache(load_train_config(matrix.config).data.cache)
    if matrix.cache_entries and len(cache) != matrix.cache_entries:
        raise ValueError(f"{cache.root} holds {len(cache)} entries, and {matrix.path.name} expects "
                         f"{matrix.cache_entries} (cache_entries): the cache is not complete, or it is another copy")
    return cache


def format_plans(plans: list[Plan], records: dict[str, RunRecord]) -> str:
    rows = [("run", "stage", "fingerprint", "images/epoch", "hours", "action")]
    total, worst = 0.0, 0.0
    for p in plans:
        by_prior, slow = estimate_hours(p, records)
        total += by_prior if by_prior is not None else slow
        worst += slow
        hours = "" if p.action not in ("train", "resume") else f"{by_prior:.1f}" if by_prior is not None else f"{slow:.1f}*"
        rows.append((p.name, p.run.stage, p.fingerprint, f"{p.images_per_epoch:,}", hours, p.describe()))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    jobs = sum(p.action in ("train", "resume") for p in plans)
    longest = max((estimate_hours(p, records)[1] for p in plans), default=0.0)
    lines += ["", f"{len(plans)} runs: {jobs} GPU jobs, {sum(p.action == 'upload' for p in plans)} uploads only, "
                  f"{sum(p.action == 'done' for p in plans)} done, {sum(p.action == 'active' for p in plans)} in the queue"]
    if jobs:
        lines += [f"estimate  about {total:.0f} GPU-hours in total, by the speed of the earlier run of each name "
                  "scaled to its new image count (* no earlier run: the slowest rate)",
                  f"          at the slowest loading measured ({SLOW_RATE} images/s): {worst:.0f} GPU-hours in total, "
                  f"{longest:.1f} hours for the longest run",
                  "          the earlier runs of one size took 0.2 to 5.5 hours, by the node, so read this as a range"]
    return "\n".join(lines)


def survey(matrix: Matrix, runs: list[Run], trial: bool, access: bool = True):
    """What preview, run, and status all read: the login, the cache, the repository, the queue, and the plans."""
    user = check_write_access(matrix.repo) if access else ""
    cache = open_cache(matrix)
    records = download_runs(matrix.repo)
    jobs = active_jobs()
    plans = make_plans(matrix, runs, cache, records, jobs or {}, trial)
    return user, cache, records, jobs, plans


def preview(matrix: Matrix, runs: list[Run], trial: bool = False, replace: bool = False,
            archive_stale: bool = False) -> bool:
    """Print what `run` would do, and change nothing. True when `run` would go ahead."""
    user, cache, records, jobs, plans = survey(matrix, runs, trial)
    print(f"login     {user} can write to {matrix.repo}")
    print(f"cache     {cache.root}: {len(cache)} entries, fingerprint {cache.header['fingerprint']}")
    print(f"config    {matrix.config}")
    print(f"code      git {git_commit() or 'unknown'}")
    print(f"jobs      {matrix.time} each, logs in {LOGS}/<run>-<job>.out\n")
    print(format_plans(plans, records))
    problems = _problems(plans, jobs, replace, archive_stale)
    for p in problems:
        print(f"problem   {p}")
    if not problems:
        print("ready     send the same arguments to `confhead train run`")
    return not problems


def _problems(plans: list[Plan], jobs: dict[str, str] | None, replace: bool, archive_stale: bool) -> list[str]:
    problems = []
    if jobs is None:
        problems.append("no squeue here: submit from a CARC login node")
    if not os.environ.get("SBATCH_ACCOUNT"):
        problems.append("SBATCH_ACCOUNT is not set: export SBATCH_ACCOUNT=<project>")
    if git_commit().endswith("-dirty"):
        problems.append("tracked files have changes: commit or undo them, so that config.json names the code that trained")
    for p in plans:
        if p.action in ("active", "done"):
            continue
        if p.local == "stale" and not archive_stale:
            problems.append(f"{p.name}: {p.dir} holds another model; --archive-stale renames the folder")
        if p.remote == "other" and not replace:
            problems.append(f"{p.name}: the repository holds another model under this name; --replace overwrites it")
    return problems


def _archive(plan: Plan) -> Path:
    saved = json.loads((plan.dir / "config.json").read_text())
    dest = plan.dir.with_name(f"{plan.name}.stale-{config_fingerprint(saved) or time.strftime('%Y%m%d-%H%M%S')}")
    if dest.exists():
        dest = dest.with_name(f"{dest.name}-{time.strftime('%Y%m%d-%H%M%S')}")
    plan.dir.rename(dest)
    return dest


def submit(matrix: Matrix, runs: list[Run], trial: bool = False, replace: bool = False, archive_stale: bool = False,
           limit: str | None = None, tag: str | None = None) -> None:
    """Upload the finished runs, submit a job for each run with epochs left, and a report job after them."""
    user, cache, records, jobs, plans = survey(matrix, runs, trial)
    problems = _problems(plans, jobs, replace, archive_stale)
    if problems:
        raise ValueError("nothing submitted:\n  " + "\n  ".join(problems))
    (PROJECT / LOGS).mkdir(parents=True, exist_ok=True)
    _sync()

    work = [p for p in plans if p.action in ("train", "resume", "upload")]
    if any(p.remote == "other" for p in work):
        made = tag_state(matrix.repo, tag or f"before-{time.strftime('%Y%m%d-%H%M%S')}")
        print(f"tag       {made}: the repository as it was, for --revision" if made
              else "tag       the current state of the repository has a tag already")
    for p in work:
        if p.local == "stale":
            print(f"archived  {p.name}: {_archive(p)}")
            p.local = "none"

    for p in [p for p in work if p.action == "upload"]:
        commit = upload_run(p.dir, matrix.repo, replace=replace)
        print(f"uploaded  {p.name}: {commit.commit_url}")

    selection = ["--repo", matrix.repo] + (["--trial"] if trial else [])
    ids = []
    for p in [p for p in work if p.action in ("train", "resume")]:
        job = _slurm("sbatch", "--parsable", f"--job-name={JOB_PREFIX}{p.name}", f"--time={limit or matrix.time}",
                     f"--output={LOGS}/{p.name}-%j.out", TRAIN_JOB, str(matrix.path), p.run.name, *selection,
                     *(["--replace"] if replace else [])).strip().split(";")[0]
        ids.append(job)
        print(f"submitted {p.name}: job {job}, {p.describe()}, log {LOGS}/{p.name}-{job}.out")
    if ids:
        report = _slurm("sbatch", "--parsable", f"--dependency=afterany:{':'.join(ids)}",
                        f"--output={LOGS}/report-%j.out", REPORT_JOB, str(matrix.path), *selection,
                        "--jobs", ",".join(ids), "--runs", *[p.run.name for p in plans]).strip().split(";")[0]
        print(f"report    job {report} starts when they end: {LOGS}/report-{report}.out")
    else:
        print("nothing to submit")


def worker(matrix: Matrix, run: Run, trial: bool = False, replace: bool = False) -> None:
    """Train one run in this process, and upload it after each epoch and at the end. The Slurm job calls it."""
    cfg = run_config(matrix, run, trial)
    training = Training(cfg)
    print(f"run       {training.dir}, fingerprint {training.fingerprint}")
    print(f"cache     {training.cache.root}: {len(training.cache)} entries, fingerprint "
          f"{training.cache.header['fingerprint']}")
    print(f"device    {training.device}, {training.num_workers} loader processes")
    try:
        there = download_runs(matrix.repo).get(cfg.output.name)
    except Exception as e:  # the check repeats at each upload
        log.warning("could not read %s: %s", matrix.repo, e)
        there = None
    if there is not None and there.fingerprint != training.fingerprint and not replace:
        raise ValueError(f"{matrix.repo} holds another model under {cfg.output.name}; submit with --replace")

    def upload(history: list[dict[str, Any]]) -> None:
        commit = upload_run(training.dir, matrix.repo, partial=len(history) < cfg.train.epochs, replace=replace)
        log.info("uploaded epoch %d of %d: %s", len(history), cfg.train.epochs, commit.commit_url)

    def upload_or_warn(history: list[dict[str, Any]]) -> None:
        try:
            upload(history)
        except Exception as e:  # an upload must never stop the training; the last one is checked below
            log.warning("upload of %s to %s failed: %s", training.dir, matrix.repo, e)

    history = training.fit(on_epoch=upload_or_warn)
    best = min(history, key=lambda h: h["dev_loss"])
    print(f"best      epoch {best['epoch']} of {len(history)}, dev loss {best['dev_loss']:.4f}: {training.dir / 'best.pt'}")
    for attempt in range(1, 4):
        try:
            upload(history)
            return
        except Exception as e:
            log.warning("final upload, attempt %d of 3, failed: %s", attempt, e)
            if attempt < 3:
                time.sleep(60)
    sys.exit(f"confhead: {cfg.output.name} is trained but not in {matrix.repo}. "
             "`confhead train run` with the same arguments uploads it from the login node")


def _accounting(job_ids: list[str]) -> dict[str, tuple[str, str, str]]:
    """Job number to (state, elapsed, exit code), from sacct."""
    try:
        out = _slurm("sacct", "-X", "--noheader", "--parsable2", "--format=JobID,State,Elapsed,ExitCode",
                     "-j", ",".join(job_ids))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    rows = (line.split("|") for line in out.splitlines() if line.strip())
    return {r[0]: (r[1].split()[0], r[2], r[3]) for r in rows if len(r) >= 4}


def status(matrix: Matrix, runs: list[Run], trial: bool = False, job_ids: list[str] | None = None) -> bool:
    """Print where each run stands, and for each job of `job_ids` that failed, the end of its log. True when all are done."""
    _, cache, records, _, plans = survey(matrix, runs, trial, access=False)
    accounting = _accounting(job_ids) if job_ids else {}
    logs = {}
    for job in job_ids or []:
        for path in (PROJECT / LOGS).glob(f"*-{job}.out"):
            logs[path.name[:-len(f"-{job}.out")]] = (job, path)

    print(f"# Training report, {time.strftime('%Y-%m-%d %H:%M')}\n")
    print(f"Repository {matrix.repo}; cache {cache.root} with {len(cache)} entries; git {git_commit() or 'unknown'}.\n")
    rows = [("run", "state", "epochs", "best dev loss", "job", "Slurm", "elapsed")]
    failures = []
    for p in plans:
        job, path = logs.get(p.name, ("", None))
        slurm, elapsed, code = accounting.get(job, ("", "", ""))
        state = {"done": "done", "active": "in the queue", "upload": "TRAINED, NOT UPLOADED",
                 "resume": "STOPPED", "train": "NOT STARTED"}[p.action]
        epochs = p.cfg.train.epochs if p.action == "done" and p.local != "complete" else p.epochs_done
        rows.append((p.name, state, f"{epochs} of {p.cfg.train.epochs}", "" if p.best is None else f"{p.best:.4f}",
                     job, slurm, elapsed))
        if p.action not in ("done", "active") or slurm in FAILED_STATES:
            failures.append((p, job, path, slurm, code))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    print("\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows))
    done = sum(p.action == "done" for p in plans)
    print(f"\n{done} of {len(plans)} runs are done and in the repository.")
    for p, job, path, slurm, code in failures:
        print(f"\n## {p.name}: {p.describe()}\n")
        if job:
            print(f"Job {job}: {slurm or 'state unknown'}, exit code {code or 'unknown'}, log {path}\n")
            lines = path.read_text(errors="replace").splitlines()[-15:]
            print("\n".join(f"    {line}" for line in lines))
        else:
            print("No job of this report trained it.")
    if failures:
        print("\nTo continue what is left, send `confhead train run` again with the same arguments.")
    return done == len(plans)


def clean_trials(matrix: Matrix) -> None:
    """Delete the trial-* run folders on disk and in the repository."""
    runs = Path(load_train_config(matrix.config).output.runs)
    for folder in sorted(runs.glob(f"{TRIAL_PREFIX}*")):
        if folder.is_dir():
            shutil.rmtree(folder)
            print(f"deleted   {folder}")
    names = [n for n in download_runs(matrix.repo) if n.startswith(TRIAL_PREFIX)]
    if names:
        commit = delete_runs(matrix.repo, names)
        print(f"deleted   {', '.join(names)} from {matrix.repo}: {commit.commit_url}")
    else:
        print(f"no trial runs in {matrix.repo}")

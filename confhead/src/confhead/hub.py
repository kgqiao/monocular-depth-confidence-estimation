"""Training runs on a Hugging Face model repository: upload, download, and the model card.

Each run is a folder of the repository named after its run folder. The model card
is built only from the config.json and history.json of each run there, so the same
runs always give the same card. An upload replaces a run of the same name only when
both have the same fingerprint, or when it is told to replace.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import CommitInfo, CommitOperationAdd, CommitOperationDelete, HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from .train import config_fingerprint

log = logging.getLogger(__name__)

REPO_ID = "csci566-depthconf/confhead"
RUN_FILES = ("best.pt", "last.pt", "config.json", "history.json")
OPTIONAL_FILES = ("steps.json",)
RECORD_FILES = ("config.json", "history.json", "steps.json")


@dataclass
class RunRecord:
    config: dict[str, Any]
    history: list[dict[str, Any]]
    steps: list[dict[str, Any]] | None = None  # absent from runs before steps.json

    @property
    def best(self) -> dict[str, Any]:
        return min(self.history, key=lambda h: h["dev_loss"])

    @property
    def epochs(self) -> int:
        return self.config["settings"]["train"]["epochs"]

    @property
    def complete(self) -> bool:
        return len(self.history) >= self.epochs

    @property
    def fingerprint(self) -> str:
        """Computed from the config, so that runs from before config.json held one have it too."""
        return config_fingerprint(self.config)

    @property
    def group(self) -> str:
        return self.config["settings"]["output"].get("group", "")

    @property
    def steps_per_epoch(self) -> int:
        train = self.config["settings"]["train"]
        return (train["samples_per_epoch"] or self.config["train_images"]) // train["batch_size"]

    @classmethod
    def read(cls, folder: Path) -> RunRecord:
        steps = folder / "steps.json"
        return cls(json.loads((folder / "config.json").read_text()), json.loads((folder / "history.json").read_text()),
                   json.loads(steps.read_text()) if steps.is_file() else None)


def download_runs(repo_id: str = REPO_ID, revision: str | None = None, api: HfApi | None = None) -> dict[str, RunRecord]:
    """The record of each run folder in the repository; empty when the repository does not exist."""
    api = api or HfApi()
    if not api.repo_exists(repo_id):
        return {}
    root = Path(snapshot_download(repo_id, revision=revision, allow_patterns=[f"*/{f}" for f in RECORD_FILES]))
    return {d.name: RunRecord.read(d) for d in sorted(root.iterdir())
            if (d / "config.json").is_file() and (d / "history.json").is_file()}


def check_write_access(repo_id: str = REPO_ID, api: HfApi | None = None) -> str:
    """The login name, once it has written to the repository; a ValueError says what is missing.

    The proof is a branch that is created and deleted at once: no role or token
    field says for certain that a write will pass.
    """
    api = api or HfApi()
    try:
        who = api.whoami()
    except (HfHubHTTPError, OSError) as e:
        raise ValueError(f"no Hugging Face login ({e}); run `uv run hf auth login`, or set $HF_TOKEN") from None
    user = who["name"]
    if not api.repo_exists(repo_id):
        raise ValueError(f"{repo_id} does not exist, or the login {user} cannot see it")
    branch = f"access-check-{user}"
    try:
        api.create_branch(repo_id, branch=branch, exist_ok=True)
        api.delete_branch(repo_id, branch=branch)
    except HfHubHTTPError as e:
        owner = repo_id.split("/")[0]
        role = next((o.get("roleInOrg") for o in who.get("orgs", []) if o.get("name") == owner), None)
        token = (who.get("auth") or {}).get("accessToken") or {}
        status = e.response.status_code if e.response is not None else "no response"
        raise ValueError(f"the login {user} cannot write to {repo_id} (HTTP {status}). Role in {owner}: "
                         f"{role or 'not a member'}; token role: {token.get('role', 'unknown')}. The role must be "
                         "write or admin, and the token a write token") from None
    return user


def tag_state(repo_id: str, tag: str, api: HfApi | None = None) -> str | None:
    """Tag the current commit of the repository, unless a tag names it already; the name of the new tag, or None.

    `hf download <repo> <file> --revision <tag>` reads the files of that commit after later uploads replace them.
    """
    api = api or HfApi()
    sha = api.repo_info(repo_id).sha
    if any(t.target_commit == sha for t in api.list_repo_refs(repo_id).tags):
        return None
    api.create_tag(repo_id, tag=tag, revision=sha)
    return tag


def delete_runs(repo_id: str, names: list[str], api: HfApi | None = None, attempts: int = 5) -> CommitInfo:
    """Delete run folders from the repository and rewrite the model card, in one commit."""
    ops = [CommitOperationDelete(f"{name}/", is_folder=True) for name in names]
    return _commit(repo_id, {}, ops, f"Delete {', '.join(names)}", api or HfApi(), attempts, drop=set(names))


def upload_run(run: str | Path, repo_id: str = REPO_ID, name: str | None = None, partial: bool = False,
               api: HfApi | None = None, attempts: int = 5, replace: bool = False) -> CommitInfo:
    """Upload a run folder and rewrite the model card, in one commit.

    The commit names the repository state the card was built from, so a commit
    that another upload got ahead of is refused, and this one reads the
    repository again and retries. A run of the same name with another
    fingerprint is another model: the upload refuses it unless `replace` is set.
    """
    run = Path(run).resolve()
    name = name or run.name
    missing = [f for f in RUN_FILES if not (run / f).is_file()]
    if missing:
        raise ValueError(f"{run} has no {', '.join(missing)}")
    record = RunRecord.read(run)
    if len(record.history) < record.epochs and not partial:
        raise ValueError(f"{run} holds {len(record.history)} of {record.epochs} epochs; finish the run, or upload it as partial")
    files = [f for f in RUN_FILES + OPTIONAL_FILES if (run / f).is_file()]
    best = record.best
    message = f"Upload {name}: epoch {len(record.history)} of {record.epochs}, best dev loss {best['dev_loss']:.4f} at epoch {best['epoch']}"
    return _commit(repo_id, {name: record}, [CommitOperationAdd(f"{name}/{f}", run / f) for f in files], message,
                   api or HfApi(), attempts, replace=replace)


def rewrite_card(repo_id: str = REPO_ID, api: HfApi | None = None, attempts: int = 5) -> CommitInfo:
    """Rewrite the model card from the runs in the repository."""
    return _commit(repo_id, {}, [], "Rewrite the model card", api or HfApi(), attempts)


def _commit(repo_id: str, new: dict[str, RunRecord], ops: list, message: str, api: HfApi, attempts: int,
            replace: bool = True, drop: frozenset[str] | set[str] = frozenset()) -> CommitInfo:
    api.create_repo(repo_id, exist_ok=True)
    for attempt in range(1, attempts + 1):
        parent = api.repo_info(repo_id).sha
        there = download_runs(repo_id, revision=parent, api=api)
        for name, record in new.items():
            if name in there and there[name].fingerprint != record.fingerprint and not replace:
                raise ValueError(f"{repo_id} holds a run {name} with another fingerprint ({there[name].fingerprint}, "
                                 f"this run has {record.fingerprint}): it is another model. Replace it on purpose, "
                                 "or use another run name")
        runs = {k: v for k, v in {**there, **new}.items() if k not in drop}
        card = CommitOperationAdd("README.md", model_card(repo_id, runs).encode())
        try:
            return api.create_commit(repo_id, ops + [card], commit_message=message, parent_commit=parent)
        except HfHubHTTPError as e:
            if e.response is None or e.response.status_code != 412 or attempt == attempts:
                raise
            log.info("%s changed during the upload; reading it again (attempt %d of %d)", repo_id, attempt + 1, attempts)
    raise AssertionError("unreachable")


CARD_HEAD = """\
---
license: cc-by-nc-sa-4.0
tags:
- depth-estimation
- nuscenes
---

# confhead

Per-pixel confidence heads for a frozen Depth Anything V2 on nuScenes, trained by
[confhead](https://github.com/kgqiao/Vision-Project-CSCI-566/tree/main/confhead).
A head predicts `p_error`, the probability that the depth of a grid cell is wrong
against lidar. Each folder is one training run:

| File | What it holds |
|---|---|
| `best.pt` | the head of the epoch with the lowest dev loss |
| `last.pt` | the head of the last epoch, with the optimizer and the schedule |
| `config.json` | every setting, the dev scenes, the class weight, the cache fingerprint, the git commit, the run fingerprint |
| `history.json` | the train and dev loss of each epoch |
| `steps.json` | the mean train loss five times per epoch, in runs that have it |

The heads train on labels from [nuScenes](https://www.nuscenes.org/), which is
under CC BY-NC-SA 4.0 and the nuScenes terms of use. The same terms apply here.

## Load a Head

```sh
hf download {repo_id} <run>/best.pt --local-dir .
```

```python
from confhead import load_head
head, inputs, fields = load_head("<run>/best.pt")
```

## Training

`confhead/src/confhead/train.py` trains every head with AdamW and a cosine learning
rate schedule over all steps, against a binary cross-entropy loss on the cells that
carry a lidar label, with the class weight below on the positive class. A cell is
positive when the absolute relative error of its depth is above `tau`. The sections
below come from the `config.json` and `history.json` of each run. Two runs with the
same run fingerprint have the same settings, cache build, train images, and dev scenes.
"""


def _num(x: float) -> str:
    return f"{x:g}"


def _depth_range(cfg: dict) -> str:
    return f"{_num(cfg['min_depth'])} to {_num(cfg['max_depth'])} m"


def _inputs(config: dict) -> str:
    on = config["settings"]["inputs"]
    groups = [g for g in ("features", "depth", "temporal", "radar") if on[g]]
    return f"{', '.join(groups)} ({len(config['channels'])} channels)"


def _summary(runs: dict[str, RunRecord]) -> list[str]:
    lines = ["## Runs", "",
             "| Run | Group | Cache | Inputs | Epochs | Best epoch | Best dev loss |",
             "|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        best = r.best
        lines.append(f"| [{name}](#{name.lower()}) | {r.group} | `{PurePosixPath(r.config['cache']['root']).name}` "
                     f"| {_inputs(r.config)} | {len(r.history)} of {r.epochs} | {best['epoch']} | {best['dev_loss']:.4f} |")
    return lines


def _run_section(name: str, r: RunRecord) -> list[str]:
    config, history = r.config, r.history
    s = config["settings"]
    inputs, train, cache, model = s["inputs"], s["train"], config["cache"], config["cache"]["model"]
    labels, radar = inputs["labels"], inputs["radar_cfg"]
    samples = train["samples_per_epoch"] or config["train_images"]
    best = r.best

    rows = [
        ("Run fingerprint", f"`{r.fingerprint}`"),
        ("Cache", f"`{PurePosixPath(cache['root']).name}`, fingerprint `{cache['fingerprint']}`, "
                  f"grid {cache['grid'][0]} × {cache['grid'][1]}"),
        ("Depth model", f"Depth Anything V2 `{model['encoder']}`, short side {model['short_side']}, "
                        f"tap `{model['tap']}`"),
        ("Inputs", _inputs(config)),
        ("Labels", f"{labels['align']}-aligned, `tau` {_num(labels['tau'])}, depth {_depth_range(labels)}"),
    ]
    if inputs["radar"]:
        sweeps = radar["max_sweeps"] or "every cached sweep"
        rows.append(("Radar", f"{radar['align']}-aligned, radius {radar['radius']} grid cells, sweeps: {sweeps}, "
                              f"depth {_depth_range(radar)}"))
    rows += [
        ("Head widths", str(s["head"]["widths"])),
        ("Epochs", str(train["epochs"])),
        ("Batch size", str(train["batch_size"])),
        ("Learning rate", _num(train["lr"])),
        ("Weight decay", _num(train["weight_decay"])),
        ("Train images per epoch", f"{samples:,} of {config['train_images']:,}"),
        ("Dev set", f"{len(config['dev_scenes'])} scenes, {config['dev_used']:,} images "
                    f"(`dev_fraction` {_num(train['dev_fraction'])}, `dev_limit` {train['dev_limit']:,})"),
        ("Positive rate", f"{config['positive_rate']:.1%} of labeled cells"),
        ("Class weight", f"{config['pos_weight']:.3f}"),
        ("Stats sample", f"{config['stats_sample']} images"),
        ("Seed", str(train["seed"])),
        ("Software", f"git `{config['git']}`, torch {config['torch']}, device `{config['device']}`"),
    ]

    lines = [f"### {name}", "",
             f"Best epoch {best['epoch']} of {len(history)}, dev loss {best['dev_loss']:.4f}. "
             f"Training took {sum(h['minutes'] for h in history) / 60:.1f} hours.", "",
             "| Setting | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += ["", "| Epoch | Train loss | Dev loss | Learning rate at end | Minutes |", "|---|---|---|---|---|"]
    for h in history:
        mark = " (best)" if h is best else ""
        lines.append(f"| {h['epoch']}{mark} | {h['train_loss']:.4f} | {h['dev_loss']:.4f} "
                     f"| {h['lr']:.2e} | {h['minutes']:.1f} |")
    return lines


def model_card(repo_id: str, runs: dict[str, RunRecord]) -> str:
    """The model card of a repository that holds `runs`, keyed by folder name."""
    runs = dict(sorted(runs.items()))
    lines = [CARD_HEAD.replace("{repo_id}", repo_id)]
    if runs:
        lines += _summary(runs)
        for name, r in runs.items():
            lines += [""] + _run_section(name, r)
    return "\n".join(lines) + "\n"

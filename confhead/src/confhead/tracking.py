"""A local Trackio project rebuilt from the run records of the Hugging Face model repository.

The model repository is the only record of a run. `sync` deletes the Trackio
project and logs every run again, so the dashboard never drifts from the repository.
Trackio is in the `tracking` dependency group: `uv run --group tracking ...`.
"""

from __future__ import annotations

from typing import Any

from .hub import RunRecord
from .train import _flat

PROJECT = "confhead"


def run_points(record: RunRecord) -> list[tuple[int, dict[str, float]]]:
    """The metrics of a run by optimizer step, in step order: the train loss of steps.json, and each epoch's end."""
    points: dict[int, dict[str, float]] = {}
    for s in record.steps or []:
        points.setdefault(s["step"], {})["train/loss"] = s["train_loss"]
    for h in record.history:
        points.setdefault(h["epoch"] * record.steps_per_epoch, {}).update({
            "epoch": h["epoch"], "train/epoch_loss": h["train_loss"], "dev/loss": h["dev_loss"],
            "lr": h["lr"], "minutes": h["minutes"]})
    return sorted(points.items())


def run_config(record: RunRecord) -> dict[str, Any]:
    """The settings of a run as flat keys, with the facts of config.json that tell runs apart."""
    c = record.config
    best = record.best
    return {**_flat(c["settings"]), "channels": len(c["channels"]), "cache.fingerprint": c["cache"]["fingerprint"],
            "train_images": c["train_images"], "dev_used": c["dev_used"], "positive_rate": c["positive_rate"],
            "pos_weight": c["pos_weight"], "git": c["git"], "device": c["device"],
            "epochs_done": len(record.history), "best_epoch": best["epoch"], "best_dev_loss": best["dev_loss"]}


def sync(runs: dict[str, RunRecord], project: str = PROJECT) -> None:
    """Replace the Trackio project with one run per record, logged in name order."""
    import trackio

    trackio.delete_project(project, force=True)
    for name, record in sorted(runs.items()):
        trackio.init(project=project, name=name, group=record.group or None, config=run_config(record),
                     embed=False, auto_log_gpu=False, auto_log_cpu=False)
        for step, metrics in run_points(record):
            trackio.log(metrics, step=step)
        trackio.finish()

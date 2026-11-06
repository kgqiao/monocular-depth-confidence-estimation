"""Upload a training run folder to a Hugging Face model repository, and rewrite its model card, in one commit.

    uv run examples/upload_run.py /scratch1/$USER/confhead-runs/run1
    uv run examples/upload_run.py /scratch1/$USER/confhead-runs/run1-no-radar --repo <user>/<name>
    uv run examples/upload_run.py                  # rewrite the model card only
    uv run examples/upload_run.py --print-card     # print the model card, upload nothing

The run goes into a folder of the repository named after the run folder, so one
repository holds every run. The model card is built only from the config.json and
history.json of each run in the repository, so the same runs always give the same
card. The files are small, so run this on the login node.
Log in once first with `uv run hf auth login`, or set $HF_TOKEN.
"""

import argparse
import json
from pathlib import Path, PurePosixPath

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO_ID = "nadayusc/confhead"
RUN_FILES = ("best.pt", "last.pt", "config.json", "history.json")

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
| `config.json` | every setting, the dev scenes, the class weight, the cache fingerprint, the git commit |
| `history.json` | the train and dev loss of each epoch |

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
below come from the `config.json` and `history.json` of each run.
"""


def _num(x: float) -> str:
    return f"{x:g}"


def _depth_range(cfg: dict) -> str:
    return f"{_num(cfg['min_depth'])} to {_num(cfg['max_depth'])} m"


def _inputs(config: dict) -> str:
    on = config["settings"]["inputs"]
    groups = [g for g in ("features", "depth", "temporal", "radar") if on[g]]
    return f"{', '.join(groups)} ({len(config['channels'])} channels)"


def _best(history: list[dict]) -> dict:
    return min(history, key=lambda h: h["dev_loss"])


def _summary(runs: dict[str, tuple[dict, list[dict]]]) -> list[str]:
    lines = ["## Runs", "",
             "| Run | Cache | Inputs | Epochs | Best epoch | Best dev loss |",
             "|---|---|---|---|---|---|"]
    for name, (config, history) in runs.items():
        best = _best(history)
        epochs = config["settings"]["train"]["epochs"]
        lines.append(f"| [{name}](#{name.lower()}) | `{PurePosixPath(config['cache']['root']).name}` "
                     f"| {_inputs(config)} | {len(history)} of {epochs} | {best['epoch']} | {best['dev_loss']:.4f} |")
    return lines


def _run_section(name: str, config: dict, history: list[dict]) -> list[str]:
    s = config["settings"]
    inputs, train, cache, model = s["inputs"], s["train"], config["cache"], config["cache"]["model"]
    labels, radar = inputs["labels"], inputs["radar_cfg"]
    samples = train["samples_per_epoch"] or config["train_images"]
    best = _best(history)

    rows = [
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


def model_card(repo_id: str, runs: dict[str, tuple[dict, list[dict]]]) -> str:
    """The model card of a repository that holds `runs`, keyed by folder name."""
    runs = dict(sorted(runs.items()))
    lines = [CARD_HEAD.replace("{repo_id}", repo_id)]
    if runs:
        lines += _summary(runs)
        for name, (config, history) in runs.items():
            lines += [""] + _run_section(name, config, history)
    return "\n".join(lines) + "\n"


def remote_runs(api: HfApi, repo_id: str) -> dict[str, tuple[dict, list[dict]]]:
    """The config and history of each run folder in the repository."""
    if not api.repo_exists(repo_id):
        return {}
    files = set(api.list_repo_files(repo_id))
    names = {f.split("/")[0] for f in files if f.count("/") == 1}
    runs = {}
    for name in sorted(n for n in names if {f"{n}/config.json", f"{n}/history.json"} <= files):
        config, history = (json.loads(Path(hf_hub_download(repo_id, f"{name}/{f}")).read_text())
                           for f in ("config.json", "history.json"))
        runs[name] = (config, history)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, nargs="?", help="the run folder, which holds best.pt; "
                                                          "omit it to rewrite the model card only")
    parser.add_argument("--repo", default=REPO_ID, help=f"the model repository (default {REPO_ID})")
    parser.add_argument("--name", help="the folder in the repository (default: the name of the run folder)")
    parser.add_argument("--partial", action="store_true", help="upload a run that has not trained every epoch")
    parser.add_argument("--print-card", action="store_true", help="print the model card and upload nothing")
    args = parser.parse_args()

    api = HfApi()
    runs = remote_runs(api, args.repo)
    ops = []
    message = "Rewrite the model card"

    if args.run:
        run = args.run.resolve()
        name = args.name or run.name
        missing = [f for f in RUN_FILES if not (run / f).is_file()]
        if missing:
            raise SystemExit(f"{run} has no {', '.join(missing)}")
        config = json.loads((run / "config.json").read_text())
        history = json.loads((run / "history.json").read_text())
        epochs = config["settings"]["train"]["epochs"]
        if len(history) < epochs and not args.partial:
            raise SystemExit(f"{run} holds {len(history)} of {epochs} epochs; finish the run, or pass --partial")
        best = _best(history)
        runs[name] = (config, history)
        ops = [CommitOperationAdd(f"{name}/{f}", run / f) for f in RUN_FILES]
        message = f"Upload {name}: best epoch {best['epoch']} of {len(history)}, dev loss {best['dev_loss']:.4f}"
        print(f"run      {run}: {len(history)} of {epochs} epochs")
        print(f"best     epoch {best['epoch']}, dev loss {best['dev_loss']:.4f}")

    card = model_card(args.repo, runs)
    if args.print_card:
        print(card, end="")
        return

    api.create_repo(args.repo, exist_ok=True)
    ops.append(CommitOperationAdd("README.md", card.encode()))
    commit = api.create_commit(args.repo, ops, commit_message=message)
    print(f"card     {len(runs)} runs: {', '.join(sorted(runs))}")
    print(f"commit   {commit.commit_url}")
    if args.run:
        print(f"files    https://huggingface.co/{args.repo}/tree/main/{name}")


if __name__ == "__main__":
    main()

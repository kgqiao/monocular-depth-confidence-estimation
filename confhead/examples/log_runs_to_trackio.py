"""Replay finished training runs into trackio, so that the runs overlay in one dashboard.

    uv run --with trackio examples/log_runs_to_trackio.py /scratch1/$USER/confhead-runs/run-v0-no-radar
    uv run --with trackio examples/log_runs_to_trackio.py --hf-repo csci566-depthconf/confhead
    uv run --with trackio examples/log_runs_to_trackio.py --hf-repo csci566-depthconf/confhead --space-id <user>/<space>

A run folder holds config.json and history.json (see guides/runs-and-configs.md).
Each run becomes one trackio run named after its folder, with one point per epoch
(train loss, dev loss, learning rate, minutes), all in one project, so the curves of
every run draw on the same charts. The settings of the run go into the trackio config,
so the dashboard can group and filter by them.

Without --space-id the runs go to the local trackio store; `trackio show` opens them.
With --space-id they go to that Hugging Face Space (created if it does not exist), which
needs a login with write access: `uv run hf auth login`, or $HF_TOKEN.
A run that is already in the project is skipped, so the script can run again.
"""

import argparse
import json
from pathlib import Path

import trackio

PROJECT = "confhead"
RUN_FILES = ("config.json", "history.json")


def flat(d: dict, prefix: str = "") -> dict:
    """{"inputs": {"radar": True}} -> {"inputs.radar": True}; lists become strings."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, f"{key}."))
        else:
            out[key] = str(v) if isinstance(v, list) else v
    return out


def summary(config: dict, history: list[dict]) -> dict:
    best = min(history, key=lambda h: h["dev_loss"])
    return {
        "channels": len(config["channels"]),
        "best_epoch": best["epoch"],
        "best_dev_loss": best["dev_loss"],
        "epochs_done": len(history),
        "class_weight": config["pos_weight"],
        "git": config["git"],
    }


def local_runs(folders: list[Path]) -> dict[str, tuple[dict, list[dict]]]:
    runs = {}
    for folder in folders:
        runs[folder.name] = tuple(json.loads((folder / f).read_text()) for f in RUN_FILES)
    return runs


def hub_runs(repo_id: str) -> dict[str, tuple[dict, list[dict]]]:
    from huggingface_hub import HfApi, hf_hub_download

    files = set(HfApi().list_repo_files(repo_id))
    names = sorted({f.split("/")[0] for f in files if "/" in f and f.endswith("history.json")})
    return {n: tuple(json.loads(Path(hf_hub_download(repo_id, f"{n}/{f}")).read_text()) for f in RUN_FILES)
            for n in names if all(f"{n}/{f}" in files for f in RUN_FILES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", type=Path, help="run folders on disk")
    parser.add_argument("--hf-repo", help="a model repository whose run folders are replayed, e.g. csci566-depthconf/confhead")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--space-id", help="a Hugging Face Space to log to; default the local store")
    parser.add_argument("--partial", action="store_true", help="also replay a run that has not trained every epoch")
    args = parser.parse_args()
    if not args.folders and not args.hf_repo:
        parser.error("give run folders or --hf-repo")

    runs = local_runs(args.folders)
    if args.hf_repo:
        runs |= hub_runs(args.hf_repo)

    known = set()
    try:
        known = {r.name for r in trackio.Api().runs(args.project)}
    except Exception:  # no such project yet, or a store that cannot be listed
        pass

    for name, (config, history) in runs.items():
        epochs = config["settings"]["train"]["epochs"]
        if name in known:
            print(f"skip     {name}: already in {args.project}")
            continue
        if len(history) < epochs and not args.partial:
            print(f"skip     {name}: {len(history)} of {epochs} epochs; finish the run, or pass --partial")
            continue
        trackio.init(project=args.project, name=name, space_id=args.space_id,
                     config={**flat(config["settings"]), **summary(config, history)})
        for h in history:
            trackio.log({"train_loss": h["train_loss"], "dev_loss": h["dev_loss"], "lr": h["lr"],
                         "minutes": h["minutes"]}, step=h["epoch"])
        trackio.finish()
        print(f"logged   {name}: {len(history)} epochs, best dev loss {min(h['dev_loss'] for h in history):.4f}")


if __name__ == "__main__":
    main()

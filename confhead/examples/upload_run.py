"""Upload a training run folder to a Hugging Face model repository, in one commit.

    uv run examples/upload_run.py /scratch1/$USER/confhead-runs/run1
    uv run examples/upload_run.py /scratch1/$USER/confhead-runs/run1-no-radar --repo <user>/<name>

The run goes into a folder of the repository named after the run folder, so one
repository holds every run. The files are small, so run this on the login node.
Log in once first with `uv run hf auth login`, or set $HF_TOKEN.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "nadayusc/confhead"
RUN_FILES = ("best.pt", "last.pt", "config.json", "history.json")

MODEL_CARD = """\
---
license: cc-by-nc-sa-4.0
tags:
- depth-estimation
- nuscenes
---

# confhead

Per-pixel confidence heads for frozen Depth Anything V2 on nuScenes, trained by
[confhead](https://github.com/kgqiao/Vision-Project-CSCI-566/tree/main/confhead).
A head predicts `p_error`, the probability that the depth of a cell is wrong
against lidar. Each folder is one training run:

| File | What it holds |
|---|---|
| `best.pt` | the head of the epoch with the lowest dev loss |
| `last.pt` | the head of the last epoch, with the optimizer and the schedule |
| `config.json` | every setting, the dev scenes, the class weight, the cache fingerprint, the git commit |
| `history.json` | the train and dev loss of each epoch |

The heads train on labels from [nuScenes](https://www.nuscenes.org/), which is
under CC BY-NC-SA 4.0 and the nuScenes terms of use. The same terms apply here.

## Load a head

    hf download {repo_id} <run>/best.pt --local-dir .

```python
from confhead import load_head
head, inputs, fields = load_head("<run>/best.pt")
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the run folder, which holds best.pt")
    parser.add_argument("--repo", default=REPO_ID, help=f"the model repository (default {REPO_ID})")
    parser.add_argument("--name", help="the folder in the repository (default: the name of the run folder)")
    parser.add_argument("--partial", action="store_true", help="upload a run that has not trained every epoch")
    args = parser.parse_args()

    run = args.run.resolve()
    name = args.name or run.name
    missing = [f for f in RUN_FILES if not (run / f).is_file()]
    if missing:
        raise SystemExit(f"{run} has no {', '.join(missing)}")

    history = json.loads((run / "history.json").read_text())
    epochs = json.loads((run / "config.json").read_text())["settings"]["train"]["epochs"]
    if len(history) < epochs and not args.partial:
        raise SystemExit(f"{run} holds {len(history)} of {epochs} epochs; finish the run, or pass --partial")
    best = min(history, key=lambda h: h["dev_loss"])

    api = HfApi()
    api.create_repo(args.repo, exist_ok=True)
    ops = [CommitOperationAdd(f"{name}/{f}", run / f) for f in RUN_FILES]
    if not api.file_exists(args.repo, "README.md"):
        ops.append(CommitOperationAdd("README.md", MODEL_CARD.replace("{repo_id}", args.repo).encode()))
    commit = api.create_commit(
        args.repo, ops,
        commit_message=f"Upload {name}: best epoch {best['epoch']} of {len(history)}, dev loss {best['dev_loss']:.4f}",
    )

    print(f"run      {run}: {len(history)} of {epochs} epochs")
    print(f"best     epoch {best['epoch']}, dev loss {best['dev_loss']:.4f}")
    print(f"commit   {commit.commit_url}")
    print(f"files    https://huggingface.co/{args.repo}/tree/main/{name}")


if __name__ == "__main__":
    main()

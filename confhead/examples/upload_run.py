"""Upload a training run folder to a Hugging Face model repository, and rewrite its model card, in one commit.

    uv run examples/upload_run.py /scratch1/$USER/confhead-runs/run1
    uv run examples/upload_run.py /scratch1/$USER/confhead-runs/run1-no-radar --repo <user>/<name>
    uv run examples/upload_run.py                  # rewrite the model card only
    uv run examples/upload_run.py --print-card     # print the model card, upload nothing

`train_head.py --upload` does the same after each epoch. The run goes into a folder
of the repository named after the run folder. The model card is built only from the
config.json and history.json of each run in the repository, so the same runs always
give the same card. Log in once first with `uv run hf auth login`, or set $HF_TOKEN.
"""

import argparse
import sys
from pathlib import Path

from confhead.hub import REPO_ID, RunRecord, download_runs, model_card, rewrite_card, upload_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, nargs="?", help="the run folder, which holds best.pt; "
                                                          "omit it to rewrite the model card only")
    parser.add_argument("--repo", default=REPO_ID, help=f"the model repository (default {REPO_ID})")
    parser.add_argument("--name", help="the folder in the repository (default: the name of the run folder)")
    parser.add_argument("--partial", action="store_true", help="upload a run that has not trained every epoch")
    parser.add_argument("--print-card", action="store_true", help="print the model card and upload nothing")
    args = parser.parse_args()

    if args.print_card:
        runs = download_runs(args.repo)
        if args.run:
            runs[args.name or args.run.resolve().name] = RunRecord.read(args.run)
        print(model_card(args.repo, runs), end="")
        return

    try:
        commit = upload_run(args.run, args.repo, args.name, args.partial) if args.run else rewrite_card(args.repo)
    except ValueError as e:
        sys.exit(f"upload_run: {e}")
    print(f"commit   {commit.commit_url}")
    if args.run:
        print(f"files    https://huggingface.co/{args.repo}/tree/main/{args.name or args.run.resolve().name}")


if __name__ == "__main__":
    main()

"""Rebuild the local Trackio project from the runs on the Hugging Face model repository, then open the dashboard.

    uv run --group tracking examples/sync_trackio.py
    uv run --group tracking examples/sync_trackio.py --no-show
    uv run --group tracking trackio show --project confhead     # with TRACKIO_DIR set as below

The Trackio database goes to outputs/trackio, or to $TRACKIO_DIR when it is set.
Each sync deletes the project and logs every run again from the repository.
"""

import argparse
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
# Trackio reads TRACKIO_DIR when it is imported.
os.environ.setdefault("TRACKIO_DIR", str(PROJECT_DIR / "outputs/trackio"))

from confhead.hub import REPO_ID, download_runs  # noqa: E402
from confhead.tracking import PROJECT, run_points, sync  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=REPO_ID, help=f"the model repository (default {REPO_ID})")
    parser.add_argument("--project", default=PROJECT, help=f"the Trackio project (default {PROJECT})")
    parser.add_argument("--no-show", action="store_true", help="rebuild the project and do not open the dashboard")
    args = parser.parse_args()

    runs = download_runs(args.repo)
    sync(runs, args.project)
    for name, r in sorted(runs.items()):
        print(f"{name:24} {r.group or '-':12} {len(r.history)} of {r.epochs} epochs, "
              f"best dev loss {r.best['dev_loss']:.4f}, {len(run_points(r))} points")
    print(f"trackio  {len(runs)} runs in project {args.project} at {os.environ['TRACKIO_DIR']}")
    if not args.no_show:
        import trackio
        trackio.show(project=args.project)


if __name__ == "__main__":
    main()

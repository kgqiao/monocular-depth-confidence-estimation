"""The confhead command.

    confhead cache info <cache folder>
    confhead cache upload <cache folder> [--config configs/huggingface.toml]
    confhead cache download <cache name> [<folder>] [--config configs/huggingface.toml]

    confhead train preview <manifest> [--stage 1 2] [--runs 'm2-*'] [--trial]
    confhead train run <manifest> [--stage 1 2] [--runs 'm2-*'] [--trial] [--replace] [--archive-stale]
    confhead train status <manifest> [--stage 1 2] [--runs 'm2-*'] [--trial]
    confhead train upload [<run folder>] [--repo <owner>/<name>]
    confhead train clean-trials <manifest>

The manifest is configs/run-matrix.toml. guides/training-on-carc.md has the steps.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BACKUP_CONFIG = PROJECT / "configs/huggingface.toml"


def _info(args: argparse.Namespace) -> None:
    from .cache.report import format_info, info
    print(format_info(info(args.cache)))


def _upload(args: argparse.Namespace) -> None:
    from .cache.backup import format_upload, load_backup_config, upload
    print(format_upload(upload(args.cache, load_backup_config(args.config))))


def _download(args: argparse.Namespace) -> None:
    from .cache.backup import download, format_download, load_backup_config
    report = download(load_backup_config(args.config), args.name, args.dest or PROJECT / "outputs/cache" / args.name)
    print(format_download(report))
    if not report.complete:
        sys.exit(1)


def _matrix(args: argparse.Namespace):
    from .matrix import load_matrix, select
    matrix = load_matrix(args.manifest)
    if getattr(args, "repo", None):
        matrix.repo = args.repo
    return matrix, select(matrix, args.stage, args.runs) if hasattr(args, "stage") else None


def _preview(args: argparse.Namespace) -> None:
    from .matrix import preview
    matrix, runs = _matrix(args)
    if not preview(matrix, runs, args.trial, args.replace, args.archive_stale):
        sys.exit(1)


def _run(args: argparse.Namespace) -> None:
    from .matrix import submit
    matrix, runs = _matrix(args)
    submit(matrix, runs, args.trial, args.replace, args.archive_stale, args.time, args.tag)


def _status(args: argparse.Namespace) -> None:
    from .matrix import status
    matrix, runs = _matrix(args)
    status(matrix, runs, args.trial, args.jobs.split(",") if args.jobs else None)


def _worker(args: argparse.Namespace) -> None:
    from .matrix import load_matrix, select, worker
    matrix = load_matrix(args.manifest)
    matrix.repo = args.repo or matrix.repo
    worker(matrix, select(matrix, patterns=[args.run_name])[0], args.trial, args.replace)


def _upload_run(args: argparse.Namespace) -> None:
    from .hub import REPO_ID, RunRecord, download_runs, model_card, rewrite_card, upload_run
    repo = args.repo or REPO_ID
    if args.print_card:
        runs = download_runs(repo)
        if args.folder:
            runs[args.name or args.folder.resolve().name] = RunRecord.read(args.folder)
        print(model_card(repo, runs), end="")
        return
    if args.folder is None:
        print(f"commit   {rewrite_card(repo).commit_url}")
        return
    commit = upload_run(args.folder, repo, args.name, args.partial, replace=args.replace)
    print(f"commit   {commit.commit_url}")
    print(f"files    https://huggingface.co/{repo}/tree/main/{args.name or args.folder.resolve().name}")


def _clean_trials(args: argparse.Namespace) -> None:
    from .matrix import clean_trials
    clean_trials(_matrix(args)[0])


def _train_parsers(train) -> None:
    def manifest(p, selection=True):
        p.add_argument("manifest", type=Path, help="the run matrix, configs/run-matrix.toml")
        p.add_argument("--repo", help="the model repository; default: the one the manifest names")
        if selection:
            p.add_argument("--stage", nargs="+", help="only the runs of these stages")
            p.add_argument("--runs", nargs="+", metavar="NAME", help="only the runs with these names; * matches any text")
            p.add_argument("--trial", action="store_true",
                           help="one short epoch of each run, under the name trial-<run>, to test the whole path")

    def flags(p):
        p.add_argument("--replace", action="store_true",
                       help="overwrite runs of the repository that have the same name and another fingerprint")
        p.add_argument("--archive-stale", action="store_true",
                       help="rename run folders on disk that hold another model under the same name")

    p = train.add_parser("preview", help="check the login, the cache, and each run; change nothing")
    manifest(p), flags(p)
    p.set_defaults(run=_preview)

    p = train.add_parser("run", help="submit a Slurm job for each run with epochs left, and upload the finished ones")
    manifest(p), flags(p)
    p.add_argument("--time", help="the time limit of each job; default: the one the manifest names")
    p.add_argument("--tag", help="the name of the tag set on the repository before a run is replaced")
    p.set_defaults(run=_run)

    p = train.add_parser("status", help="where each run stands: on disk, in the repository, in the queue")
    manifest(p)
    p.add_argument("--jobs", help="job numbers, with commas: add their Slurm state and the log end of each failure")
    p.set_defaults(run=_status)

    p = train.add_parser("worker", help="train one run in this process; the Slurm job calls it")
    p.add_argument("manifest", type=Path)
    p.add_argument("run_name")
    p.add_argument("--repo")
    p.add_argument("--trial", action="store_true")
    p.add_argument("--replace", action="store_true")
    p.set_defaults(run=_worker)

    p = train.add_parser("upload", help="upload a run folder by hand, or rewrite the model card when none is given")
    p.add_argument("folder", type=Path, nargs="?", help="the run folder, which holds best.pt")
    p.add_argument("--repo", help="the model repository")
    p.add_argument("--name", help="the folder in the repository; default: the name of the run folder")
    p.add_argument("--partial", action="store_true", help="upload a run that has not trained every epoch")
    p.add_argument("--replace", action="store_true", help="overwrite a run of the same name and another fingerprint")
    p.add_argument("--print-card", action="store_true", help="print the model card and upload nothing")
    p.set_defaults(run=_upload_run)

    p = train.add_parser("clean-trials", help="delete the trial-* runs on disk and in the repository")
    manifest(p, selection=False)
    p.set_defaults(run=_clean_trials)


def main() -> None:
    parser = argparse.ArgumentParser(prog="confhead", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    groups = parser.add_subparsers(required=True, metavar="command")
    cache = groups.add_parser("cache", help="look at, back up, and restore a cache").add_subparsers(
        required=True, metavar="action")

    p = cache.add_parser("info", help="what a cache folder holds, and its size on disk")
    p.add_argument("cache", type=Path, help="the cache folder, which holds cache.json")
    p.set_defaults(run=_info)

    p = cache.add_parser("upload", help="back up a cache to the Hugging Face repository of the config")
    p.add_argument("cache", type=Path, help="the cache folder, which holds cache.json")
    p.add_argument("--config", type=Path, default=BACKUP_CONFIG, help="default %(default)s")
    p.set_defaults(run=_upload)

    p = cache.add_parser("download", help="restore a cache from the Hugging Face repository of the config")
    p.add_argument("name", help="the name of the cache folder in the repository, e.g. nuscenes-trainval")
    p.add_argument("dest", type=Path, nargs="?", help="the folder to restore into; default outputs/cache/<name>")
    p.add_argument("--config", type=Path, default=BACKUP_CONFIG, help="default %(default)s")
    p.set_defaults(run=_download)

    _train_parsers(groups.add_parser("train", help="train the runs of the run matrix on CARC").add_subparsers(
        required=True, metavar="action"))

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        args.run(args)
    except (FileNotFoundError, ValueError, OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"confhead: {e}")


if __name__ == "__main__":
    main()

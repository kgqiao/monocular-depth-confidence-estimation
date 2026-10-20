"""The confhead command.

    confhead cache info <cache folder>
    confhead cache upload <cache folder> [--config configs/huggingface.toml]
    confhead cache download <cache name> [<folder>] [--config configs/huggingface.toml]
"""

import argparse
import logging
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

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        args.run(args)
    except (FileNotFoundError, ValueError, OSError) as e:
        sys.exit(f"confhead: {e}")


if __name__ == "__main__":
    main()

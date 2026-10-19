"""Command line entry point: `confcache <command>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, to_toml

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd()
DEFAULT_CONFIG = BASE_DIR / "configs" / "nuscenes-mini.toml"


def parse_shard(text: str) -> tuple[int, int]:
    i, sep, n = text.partition("/")
    if not sep or not i.isdigit() or not n.isdigit() or not 0 <= int(i) < int(n):
        raise argparse.ArgumentTypeError(f"--shard {text!r}: write it as i/n with 0 <= i < n")
    return int(i), int(n)


def config_of(args):
    path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None)
    try:
        return load_config(path, args.set)
    except ValueError as e:
        raise SystemExit(f"config: {e}")


def root_of(args) -> Path:
    return Path(args.root) if args.root else Path(config_of(args).output.root)


def cmd_plan(args) -> None:
    from .report import format_plan, plan

    cfg = config_of(args)
    print(format_plan(plan(cfg, args.shard), cfg))


def cmd_config(args) -> None:
    cfg = config_of(args)
    print(f"# fingerprint {cfg.fingerprint()}")
    print(to_toml(cfg), end="")


def cmd_build(args) -> None:
    from .build import build

    report = build(config_of(args), shard=args.shard, limit=args.limit, overwrite=args.overwrite)
    if report.failed:
        sys.exit(1)


def cmd_index(args) -> None:
    from .store import build_index

    cfg = config_of(args)
    root = Path(args.root) if args.root else Path(cfg.output.root)
    index = build_index(root, cfg.splits if args.config or not args.root else None)
    tags = {}
    for row in index["entries"]:
        for t in row["tags"]:
            tags[t] = tags.get(t, 0) + 1
    print(f"indexed {len(index['entries'])} entries in {root / 'index.json'} "
          f"({index['stale_entries']} stale left out); tags: {tags or 'none'}")


def cmd_verify(args) -> None:
    from .derive import LabelConfig
    from .report import format_verify, verify, write_summary

    root = root_of(args)
    result = verify(root, LabelConfig(tau=args.tau))
    print(format_verify(result))
    if args.summary:
        write_summary(result, (root / "config.toml").read_text(), Path(args.summary))
        print(f"wrote the summary to {args.summary}")
    if result["problems"]:
        sys.exit(1)


def cmd_preview(args) -> None:
    from .derive import LabelConfig
    from .report import preview
    from .store import Cache

    root = root_of(args)
    cache = Cache(root)
    keys = args.keys
    if not keys:
        rows = cache.rows()
        keys = [r["key"] for r in rows[:: max(1, len(rows) // args.count)][: args.count]]
    out = Path(args.out) if args.out else root / "previews"
    for key in keys:
        print(preview(cache, key, args.dataroot, out, LabelConfig(tau=args.tau)))


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help=f"TOML settings; default {DEFAULT_CONFIG.relative_to(BASE_DIR)}")
    common.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="change one setting; repeat for more")

    parser = argparse.ArgumentParser(prog="confcache", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("config", parents=[common], help="print the settings after the file and --set")
    p.set_defaults(run=cmd_config)

    p = sub.add_parser("plan", parents=[common], help="count the items and estimate the disk size")
    p.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="I/N")
    p.set_defaults(run=cmd_plan)

    p = sub.add_parser("build", parents=[common], help="build or continue the cache")
    p.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="I/N",
                   help="build every N-th item from I, for one job of a job array")
    p.add_argument("--limit", type=int, default=0, help="build at most this many items, for a trial")
    p.add_argument("--overwrite", action="store_true",
                   help="build every entry again, also when the cache holds other settings")
    p.set_defaults(run=cmd_build)

    p = sub.add_parser("index", parents=[common], help="write index.json again, with the splits of the config")
    p.add_argument("--root", help="cache folder; default output.root of the config")
    p.set_defaults(run=cmd_index)

    p = sub.add_parser("verify", parents=[common], help="check every entry and print statistics")
    p.add_argument("--root", help="cache folder; default output.root of the config")
    p.add_argument("--tau", type=float, default=0.10, help="AbsRel threshold of the label statistics")
    p.add_argument("--summary", help="also write the result as markdown to this file")
    p.set_defaults(run=cmd_verify)

    p = sub.add_parser("preview", parents=[common], help="draw entries as image panels")
    p.add_argument("keys", nargs="*", help="entry keys; default a spread of --count entries")
    p.add_argument("--root", help="cache folder; default output.root of the config")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--out", help="default <root>/previews")
    p.add_argument("--dataroot", help="nuScenes folder; default the one the cache was built from")
    p.add_argument("--tau", type=float, default=0.10)
    p.set_defaults(run=cmd_preview)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.run(args)

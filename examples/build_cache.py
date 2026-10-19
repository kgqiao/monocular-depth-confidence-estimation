"""Plan, build, and check the cache of frozen Depth Anything V2 outputs.

    uv run examples/build_cache.py --plan
    uv run examples/build_cache.py --limit 12
    uv run examples/build_cache.py --set grid.patch_multiple=2 --set output.root=outputs/cache/mini-g2
    uv run examples/build_cache.py --shard 0/8           # one task of a job array

A stopped build continues where it stopped. After the shards of a job array
end, run it once more without --shard: every entry is done, so it only
writes the index and checks the cache.
"""

import argparse
import logging
import sys
from pathlib import Path

from confhead.cache import build, format_plan, format_verify, load_config, plan, to_toml, verify

REPO = Path(__file__).resolve().parents[1]


def shard(text: str) -> tuple[int, int]:
    i, _, n = text.partition("/")
    if not (i.isdigit() and n.isdigit() and 0 <= int(i) < int(n)):
        raise argparse.ArgumentTypeError(f"--shard {text!r}: write it as i/n with 0 <= i < n")
    return int(i), int(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=REPO / "configs/nuscenes-mini.toml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="change one setting; repeat for more")
    parser.add_argument("--plan", action="store_true", help="print the plan and the settings, and stop")
    parser.add_argument("--shard", type=shard, default=(0, 1), metavar="I/N")
    parser.add_argument("--limit", type=int, default=0, help="build at most this many entries")
    parser.add_argument("--overwrite", action="store_true", help="build every entry again, also with other settings")
    parser.add_argument("--summary", type=Path, help="also write the check as markdown to this file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config, args.set)
    if args.plan:
        print(format_plan(plan(cfg, args.shard), cfg))
        print(f"\n# fingerprint {cfg.fingerprint()}\n{to_toml(cfg)}", end="")
        return

    report = build(cfg, shard=args.shard, limit=args.limit, overwrite=args.overwrite)
    if args.shard != (0, 1):
        sys.exit(1 if report.failed else 0)

    result = verify(cfg.output.root)
    text = format_verify(result)
    print(text)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        settings = (Path(cfg.output.root) / "config.toml").read_text()
        summary = (f"# Cache summary\n\nWritten by `examples/build_cache.py`.\n\n```\n{text}\n```\n\n"
                   f"Settings:\n\n```toml\n{settings}```\n")
        args.summary.write_text(summary.replace(f"{REPO}/", ""))
    sys.exit(1 if report.failed or result["problems"] else 0)


if __name__ == "__main__":
    main()

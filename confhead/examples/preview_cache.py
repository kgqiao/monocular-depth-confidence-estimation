"""Draw cache entries as figures: image and lidar, predicted depth, temporal spread, features, labels, radar.

    uv run examples/preview_cache.py scene-0553_020_CAM_FRONT
    uv run examples/preview_cache.py --count 12 --cache outputs/cache/nuscenes-mini-g2
"""

import argparse
from pathlib import Path

from confhead.cache import Cache
from confhead.labels import LabelConfig
from confhead.plot import entry_figure

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keys", nargs="*", help="entry keys; default a spread of --count entries")
    parser.add_argument("--cache", type=Path, default=PROJECT / "outputs/cache/nuscenes-mini")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--dataroot", type=Path, help="nuScenes folder; default the one the cache was built from")
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--out", type=Path, help="default <cache>/previews")
    args = parser.parse_args()

    cache = Cache(args.cache, args.dataroot)
    keys = args.keys
    if not keys:
        rows = cache.rows()
        keys = [r["key"] for r in rows[:: max(1, len(rows) // args.count)][: args.count]]
    out = args.out or args.cache / "previews"
    out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        path = out / f"{key}.png"
        entry_figure(cache.entry(key), LabelConfig(tau=args.tau)).savefig(path, dpi=110)
        print(path)


if __name__ == "__main__":
    main()

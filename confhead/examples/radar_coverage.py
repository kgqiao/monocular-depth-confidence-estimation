"""Measure how much of each cached camera image the radar channels cover.

    uv run examples/radar_coverage.py --cache /scratch1/$USER/confhead-cache/nuscenes-trainval

For every entry it counts the radar points, the grid cells they mark as valid
at radius 0, 1 (the training default) and 2, and the share of the labeled
lidar cells that sit under a valid radar cell. It writes one row per entry to a
CSV and the summary, by split, condition, weather and camera, to a Markdown file.
"""

import argparse
import csv
import statistics
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from confhead.cache.store import Cache
from confhead.dataset import RadarConfig
from confhead.geometry import rasterize
from confhead.labels import LabelConfig, fit_of, lidar_labels

PROJECT = Path(__file__).resolve().parents[1]
RADII = (0, 1, 2)
_cache: Cache | None = None


def _init(root: str) -> None:
    global _cache
    _cache = Cache(root)


def measure(row: dict) -> dict:
    entry = _cache.entry(row)
    meta = entry.meta
    a = entry.load(["radar", "lidar"])
    grid = tuple(meta["grid_size"])
    image = (meta["image"]["width"], meta["image"]["height"])
    cells = grid[0] * grid[1]
    pts = a["radar"]
    out = {
        "key": row["key"], "scene": row["scene"], "channel": row["channel"],
        "condition": row["condition"], "weather": row["weather"], "tags": "+".join(row["tags"]),
        "radar_points": len(pts),
        "radar_points_keyframe": int((pts[:, 4] < 1).sum()) if len(pts) else 0,
        "radar_fit": int(fit_of(meta, "radar") is not None),
    }
    masks = {}
    for r in RADII:
        _, masks[r] = rasterize(pts, np.zeros(len(pts)), image, grid, radius=r)
        out[f"valid_r{r}"] = float(masks[r].sum() / cells)
    keyframe = pts[pts[:, 4] < 1] if len(pts) else pts
    _, key_mask = rasterize(keyframe, np.zeros(len(keyframe)), image, grid, radius=RadarConfig().radius)
    out["valid_r1_keyframe"] = float(key_mask.sum() / cells)
    _, _, labeled = lidar_labels(meta, a, LabelConfig())
    n = int(labeled.sum())
    out["label_cells"] = n / cells
    out["labeled_under_radar_r1"] = float((labeled & masks[1]).sum() / n) if n else float("nan")
    return out


def summarize(rows: list[dict]) -> str:
    def stat(g: list[dict], field: str, fn) -> float:
        return fn([r[field] for r in g if r[field] == r[field]])

    lines = ["# Radar coverage of the cache", "",
             "Written by `examples/radar_coverage.py`. One row per cache entry (a camera image). "
             "Valid cells are the cells the radar validity channel marks, as a share of the grid. "
             "`r1` is the training default (`radius` 1, every cached sweep). "
             "`Labeled under radar` is the share of lidar-labeled cells that have a valid radar cell on them.", ""]
    for title, key in (("Split", "tags"), ("Condition", "condition"), ("Weather", "weather"), ("Camera", "channel")):
        groups = defaultdict(list)
        for r in rows:
            for g in (r[key].split("+") if key == "tags" else [r[key]]):
                groups[g].append(r)
        groups["all"] = rows
        lines += [f"## By {title.lower()}", "",
                  f"| {title} | Images | Radar points (mean) | Valid cells r0 | r1 | r2 | r1, keyframe sweep only "
                  "| Images with no radar | Labeled under radar (r1) | Radar fit failed |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name in sorted(groups, key=lambda k: (k == "all", k)):
            g = groups[name]
            lines.append(
                f"| {name} | {len(g)} | {stat(g, 'radar_points', statistics.fmean):.1f} "
                f"| {stat(g, 'valid_r0', statistics.fmean):.2%} | {stat(g, 'valid_r1', statistics.fmean):.2%} "
                f"| {stat(g, 'valid_r2', statistics.fmean):.2%} | {stat(g, 'valid_r1_keyframe', statistics.fmean):.2%} "
                f"| {sum(r['radar_points'] == 0 for r in g) / len(g):.1%} "
                f"| {stat(g, 'labeled_under_radar_r1', statistics.fmean):.2%} "
                f"| {sum(r['radar_fit'] == 0 for r in g) / len(g):.1%} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=PROJECT / "outputs/radar_coverage.csv")
    parser.add_argument("--summary", type=Path, default=PROJECT / "outputs/radar-coverage.md")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="measure only the first N entries, to test")
    args = parser.parse_args()

    cache = Cache(args.cache)
    todo = cache.rows()[: args.limit or None]
    print(f"{len(todo)} entries, {args.workers} workers", flush=True)
    with Pool(args.workers, initializer=_init, initargs=(str(args.cache),)) as pool:
        rows = []
        for i, r in enumerate(pool.imap(measure, todo, chunksize=64), 1):
            rows.append(r)
            if i % 10000 == 0:
                print(f"{i}/{len(todo)}", flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summarize(rows) + "\n")
    print(f"wrote {args.csv} and {args.summary}")


if __name__ == "__main__":
    main()

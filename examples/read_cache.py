"""Read a cache as the confidence head will train on it, and print what one batch holds.

    uv run examples/read_cache.py --cache outputs/cache/nuscenes-mini-g2
"""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from confhead import Cache, ConfidenceDataset, HeadInputs

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=REPO / "outputs/cache/nuscenes-mini")
    args = parser.parse_args()

    cache = Cache(args.cache)
    for tag in ("train", "val"):
        rows = cache.rows(tag=tag)
        scenes = sorted({r["scene"] for r in rows})
        print(f"{tag:6} {len(rows)} entries from {len(scenes)} scenes")

    train = ConfidenceDataset(cache, tag="train")
    batch = next(iter(DataLoader(train, batch_size=8, shuffle=True, num_workers=2)))
    names = train.inputs.channel_names(batch["x"].shape[1] - 4)
    print(f"x       {tuple(batch['x'].shape)}: {names[0]} .. {names[-5]}, then {', '.join(names[-4:])}")
    labeled = batch["mask"].sum()
    print(f"mask    {tuple(batch['mask'].shape)}: {labeled / batch['mask'].numel():.1%} of cells hold a lidar label")
    print(f"target  {tuple(batch['target'].shape)}: {batch['target'].sum() / labeled:.1%} of labeled cells are wrong")

    no_radar = ConfidenceDataset(cache, HeadInputs(radar=False), tag="train")
    print(f"without radar, x has {no_radar[0]['x'].shape[0]} channels")


if __name__ == "__main__":
    main()

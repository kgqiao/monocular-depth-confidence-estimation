"""Train the confidence head on a cache, and write its run folder.

    uv run examples/train_head.py --estimate
    uv run examples/train_head.py
    uv run examples/train_head.py --config configs/train-trainval.toml
    uv run examples/train_head.py --set inputs.radar=false --set output.name=mini-no-radar

--estimate trains nothing: it prints the splits, the input statistics, and the
time of one epoch. A run folder that holds last.pt continues after its last
epoch, so the same command finishes a run that stopped.
"""

import argparse
import logging
import sys
from pathlib import Path

from confhead.train import POSITIVE_RATE_GATE, Training, load_train_config

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=PROJECT / "configs/train-mini.toml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="change one setting; repeat for more")
    parser.add_argument("--estimate", action="store_true",
                        help="print the splits, the input statistics, and the time of one epoch, and stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        training = Training(load_train_config(args.config, args.set))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"train_head: {e}")
    cfg = training.cfg
    print(f"cache    {training.cache.root}: {len(training.cache)} entries, "
          f"fingerprint {training.cache.header['fingerprint']}")
    print(f"run      {training.dir}")
    print(f"device   {training.device}, {training.num_workers} loader processes")
    for s in training.splits():
        print(f"{s['split']:8} {s['scenes']} scenes, {s['images']} images, {s['used']} used; "
              f"{s['night_scenes']} night and {s['rain_scenes']} rain scenes")

    if args.estimate:
        stats = training.input_stats()
        gate = "inside" if stats.in_gate else "OUTSIDE, see the Run 1 plan"
        print(f"labels   {stats.labeled_fraction:.1%} of cells labeled; positive rate {stats.positive_rate:.1%} "
              f"(gate {POSITIVE_RATE_GATE[0]:.0%} to {POSITIVE_RATE_GATE[1]:.0%}: {gate}); "
              f"class weight {stats.pos_weight:.3f}")
        speed = training.speed()
        minutes, total = speed.minutes_per_epoch, cfg.train.epochs * speed.minutes_per_epoch
        print(f"speed    loading {speed.load_rate:.0f} images/s, training step {speed.step_rate:.0f} images/s")
        print(f"epoch    {speed.images_per_epoch} images in about {minutes:.1f} min; {cfg.train.epochs} epochs in about "
              + (f"{total:.0f} min" if total < 120 else f"{total / 60:.1f} h"))
        if minutes > 120:
            print("         over 2 hours per epoch: set train.samples_per_epoch to half the train images")
        return

    try:
        history = training.fit()
    except ValueError as e:
        sys.exit(f"train_head: {e}")
    best = min(history, key=lambda h: h["dev_loss"])
    print(f"best     epoch {best['epoch']} of {len(history)}, dev loss {best['dev_loss']:.4f}: {training.dir / 'best.pt'}")


if __name__ == "__main__":
    main()

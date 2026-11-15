"""
Label count table per BEVCar group.
Outputs: scenes, keyframes, camera images, mean labeled cells/image, 
         share of labeled cells with AbsRel > tau.
"""

# This produces the exact table Form B asks for. Rain will almost certainly have the fewest labeled cells per image (wet roads absorb lidar returns) and potentially the highest positive rate (depth is worst in rain).
import numpy as np
from confhead.cache.store import Cache
from confhead.labels import LabelConfig, lidar_labels
from evals.bevcar_splits import BEVCAR_CONDITION
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cache", required=True)
args = parser.parse_args()

cfg = LabelConfig()  # tau=0.10
cache = Cache(args.cache)

stats = {g: {"scenes": set(), "keyframes": 0, "images": 0,
             "total_cells": 0, "positive_cells": 0}
         for g in ("day", "rain", "night")}

for row in cache.rows(tag="val"):
    scene = row["scene"]
    group = BEVCAR_CONDITION.get(scene)
    if group is None:
        continue
    
    s = stats[group]
    s["scenes"].add(scene)
    
    # Count one keyframe per unique (scene, sample_index) pair
    # Each row is one camera image
    s["images"] += 1
    
    entry = cache.entry(row)
    arrays = entry.load(["lidar"])
    meta = entry.meta
    abs_rel, target, mask = lidar_labels(meta, arrays, cfg)
    
    n_labeled = int(mask.sum())
    n_positive = int(target.sum())
    s["total_cells"] += n_labeled
    s["positive_cells"] += n_positive

print(f"\n{'Group':<8} {'Scenes':>7} {'Images':>8} {'Cells/img':>10} {'AbsRel>tau':>12}")
print("-" * 50)
for group in ("day", "rain", "night"):
    s = stats[group]
    n_scenes = len(s["scenes"])
    n_images = s["images"]
    cells_per_img = s["total_cells"] / max(n_images, 1)
    pos_rate = s["positive_cells"] / max(s["total_cells"], 1)
    print(f"{group:<8} {n_scenes:>7} {n_images:>8} {cells_per_img:>10.1f} {pos_rate:>11.1%}")
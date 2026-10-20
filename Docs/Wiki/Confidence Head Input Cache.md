---
id: doc-kj6mm4
type: topic
kind: concept
description: "The on-disk cache of frozen Depth Anything V2 outputs per nuScenes camera image: decoder features, disparity, temporal spread, lidar and radar points, and scale-and-shift fits."
tags: [machine-learning/cs566-deeplearning/cs566-project/kqiao-cs566-project-repo]
aliases: [confhead cache, Depth Anything V2 feature cache]
created: 2026-10-19T17:34:12
updated: 2026-10-19T17:35:36
refreshed: 2026-10-19T17:35:36
status: stable
sources: ["[[kqiao-cs566-project-repo @ ff2bc31]]", "[[CS566 Brainstorm - Project Scope Visual (Form B)]]", "[[CS566 Project Pre-proposal]]"]
up: ["[[confhead library]]", "[[Per-Pixel Depth Confidence Head]]"]
---

> [!concept] Concept
> 3 sources · refreshed 2026-10-19 · #machine-learning/cs566-deeplearning/cs566-project/kqiao-cs566-project-repo
> Up: [[confhead library]] · [[Per-Pixel Depth Confidence Head]]

## Definition

The input cache holds, for every camera image of every nuScenes keyframe, what costs a forward pass of the frozen [[Depth Anything V2]] or needs the raw dataset. The [[Per-Pixel Depth Confidence Head]] then trains on arrays, not images. This is the cache pass of Form B's training path ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Training Path) and the caching step of the pre-proposal ([[CS566 Project Pre-proposal]], p. 2, §2.1). The [[confhead library]] builds and reads it (`confhead/src/confhead/cache/`).

## Explanation

### What an entry holds

One `.npz` of arrays and one `.json` that says what they mean, per camera image. The grid is the ViT patch grid times `grid.patch_multiple`: 4 by default (264 x 148), 2 for the local mini build (132 x 74) (`confhead/src/confhead/cache/config.py:45`).

| Array | Shape | Type | Content |
|---|---|---|---|
| `features` | (C, h, w) | float16 | Decoder features at the tap, area-averaged to the grid. Default tap `output_conv1`, 64 channels; `refinenet1` and `refinenet2` give 128. |
| `disparity_grid` | (h, w) | float32 | The model's relative disparity on the grid |
| `temporal_relstd` | (h, w) | float16 | [[Ego-Motion-Compensated Temporal Spread]] |
| `temporal_count` | (h, w) | uint8 | Frames that see the cell, the keyframe included |
| `lidar`, `radar` | (N, 6) | float32 | Points as rows of (u, v, z, dt, sweep, disparity) in original image pixels |

The `.json` holds the calibration, the pose, the condition and weather, and two least-squares fits of disparity to inverse depth: one to the keyframe lidar, one to the radar. A fit with a scale that is not positive is refused.

The labels, the radar residual, and the radar validity are not stored. They are computed when an entry loads, so `tau`, the alignment, and the radar radius change without a rebuild (`confhead/src/confhead/labels.py`, `confhead/src/confhead/dataset.py`).

### How it is built

- Worker processes read the images and the point files. The main process runs each keyframe image and its neighbor images through the model in batches, derives the grid maps, and hands each entry to writer threads.
- The settings that change what an entry holds make a fingerprint. An entry with the current fingerprint is not built again, so a stopped build continues where it stopped. A build into a cache with other settings stops and names the settings that differ. The checkpoint file name is part of the fingerprint, so even re-indexing needs `DEPTH_ANYTHING_CHECKPOINT` set.
- `--shard i/n` builds every n-th item, for one task of a CARC job array.
- Each entry's `.json` is written after its `.npz`, so it marks a complete entry.

### Measured on nuScenes v1.0-mini

At `grid.patch_multiple = 2`, with the official mini split (8 train scenes, 2 val), at `ff2bc31` (`confhead/results/cache-nuscenes-mini-g2.md`; [[2026-10-18 2158 f8be68]]):

| Measure | Value |
|---|---|
| Entries | 2,424 (1,938 train, 486 val); 0 failed |
| Size | 3.43 GB, 1.4 MB per entry; the default grid gives 5.4 MB per entry, 13 GB in all |
| Build time | 42 min on an Apple GPU, about 0.96 images per second, 5 model passes per image |
| Points per image | 3,613 lidar (keyframe sweep), 150 radar (up to 5 sweeps per radar) |
| Fits | Lidar fit on 99.8% of images, radar fit on 86.5%; mean lidar fit RMSE 0.024 1/m |
| Labels | 26.2% of grid cells hold a lidar label; 37.2% of those have AbsRel > 0.10 |
| AbsRel | Median over images of the per-image median: day 0.052, night 0.079 |
| Temporal spread | 99.5% of cells seen by 2 or more frames; median 0.021 |

The 4 images with no lidar fit are night images of scene-1077: their disparity does not correlate with the lidar (correlation -0.15 and -0.02), so they have no labels and no temporal spread.

### Reading it

`ConfidenceDataset(Cache(root), tag="train")` gives `x` of shape (C + 4, h, w): the features, the disparity over its image mean, the temporal spread, the radar residual, and the radar validity. It also gives `target` (1 where AbsRel > tau), `mask` (1 where a lidar label exists), and `abs_rel`. `HeadInputs(radar=False)` and the like switch channel groups off for the ablations.

## Related

- [[confhead library]]: the library that builds and reads the cache.
- [[Ego-Motion-Compensated Temporal Spread]]: the temporal channel.
- [[Lidar and Radar Projection into Camera Images]]: the projection of the stored points.
- [[CS566 Confidence Head Training Run 1]]: the plan, which keeps no feature store.
- [[CS566 Implementation Deviations and Open Questions]]: where the cache departs from the plans.

## Sources

- [[kqiao-cs566-project-repo @ ff2bc31]]
- [[CS566 Brainstorm - Project Scope Visual (Form B)]]
- [[CS566 Project Pre-proposal]]

# The proposal against the confhead pipeline

What the [pre-proposal](Prepreposal/Deliberable/CS566%20-%20Project%20Pre-proposal.pdf) and [Form B](Prepreposal/Brainstorm/Project%20Scope%20Visual%20(Form%20B).md) ask for, and what [`confhead`](../confhead/README.md) does. Checked on 2026-10-19 against the code before the full nuScenes cache build.

## What matches

| Proposal says | Pipeline does | Where |
|---|---|---|
| Inputs: decoder features (C ≈ 64–128), predicted depth, temporal variance, radar residual, radar validity | `features` (64 channels from `output_conv1`), `disparity_grid`, `temporal_relstd`, and the radar point list. `ConfidenceDataset` stacks them as `x` with C + 4 channels | `cache/build.py:derive`, `dataset.py` |
| Temporal variance over ±2 frames at 12 Hz, ego-motion compensated | `offsets = [-2, -1, 1, 2]`, lifted through the keyframe depth and both camera poses | `cache/temporal.py` |
| Label: AbsRel > 0.10 against lidar, after a per-image scale-and-shift fit | `LabelConfig(tau=0.10)`, fit to the keyframe lidar sweep, refused when the scale is not positive | `labels.py`, `geometry.py:fit_scale_shift` |
| Masked BCE at pixels with a lidar return | `target` and `mask` per entry. A failed fit gives an empty mask, so the image adds nothing to the loss | `dataset.py` |
| Sparsification curve needs the true error per pixel | `abs_rel` is returned with every sample | `dataset.py` |
| Raw temporal variance as a baseline | A cached channel, so it costs nothing | `cache/build.py` |
| Ablations without radar, or on features alone | `HeadInputs(features=..., radar=...)` switches channel groups off at load time | `dataset.py` |
| Report by day, night, rain on the official val split | `[splits]` holds the devkit's 700/150 split. The index tags each entry with `condition` and `weather` | `configs/nuscenes-trainval.toml`, `cache/store.py` |
| Depth range 1 m to 80 m | `points.min_depth = 1.0`, `points.max_depth = 80.0` | `cache/config.py` |

## What differs

| Proposal says | Pipeline does | What to do |
|---|---|---|
| Flip-augmentation disagreement as a baseline | Nothing in the cache stores a pass on the flipped image | Decide before the full build. Adding it costs one more model pass per item (about 20% more build time) and about 3% more disk. It changes the fingerprint, so it cannot be added to a built cache. |
| MC dropout as a baseline, where the checkpoint permits | The DA-V2 checkpoint has no dropout layers | Likely out. The cache cannot help it. |
| One confidence channel at image resolution | Everything is on the 264 x 148 grid at `patch_multiple = 4`. One cell covers about 6 x 6 pixels. Labels are rasterized to the grid; the nearest lidar point wins per cell | The head predicts on the grid. Upsample the map for figures, or state the grid in the report. AUSE and AURG are computed over labeled cells either way. |
| Weather split from the BEVCar release of the nuScenes val split | `condition` is `night` when the scene description contains "night"; `weather` is `rain` when it contains "rain" | Every index row holds the scene name, so a BEVCar split applies at evaluation time with no rebuild. Check that release before the results are written. |
| Under one percent of pixels carry a label | About 26% of grid cells at `patch_multiple = 2`; fewer at 4 | Both are true and count different things. Report the grid number: that is what the head trains on. |
| Radar residual: predicted depth against radar range | The residual uses a scale-and-shift fit to the same radar points it measures (`RadarConfig.align = "radar"`), because a deployed car has no lidar | A within-image consistency signal, not an independent check. Both fits are in every entry's meta, so `RadarConfig(align="lidar")` is a one-line ablation. |

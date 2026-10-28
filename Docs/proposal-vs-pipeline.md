# The proposal against the confhead pipeline

What the [pre-proposal](Prepreposal/Deliberable/CS566%20-%20Project%20Pre-proposal.pdf) and [Form B](Prepreposal/Brainstorm/Project%20Scope%20Visual%20(Form%20B).md) ask for, and what [`confhead`](../confhead/README.md) does. First checked on 2026-10-19, before the full nuScenes cache build. Reviewed item by item on 2026-10-28, after the build: each item under What differs has an action in [training-next-actions.md](training-next-actions.md), and the others are accepted with a reason.

## What matches

| Proposal says | Pipeline does | Where |
|---|---|---|
| Inputs: decoder features (C ≈ 64–128), predicted depth, temporal variance, radar residual, radar validity | `features` (64 channels from `output_conv1`), `disparity_grid` (the relative disparity over its image mean, with no fit), `temporal_relstd`, and the radar point list. `ConfidenceDataset` stacks them as `x` with C + 4 channels | `cache/build.py:derive`, `dataset.py` |
| Temporal variance over ±2 frames at 12 Hz, ego-motion compensated | `offsets = [-2, -1, 1, 2]`, lifted through the keyframe depth and both camera poses | `cache/temporal.py` |
| Label: AbsRel > 0.10 against lidar, after a per-image scale-and-shift fit | `LabelConfig(tau=0.10)`, fit to the keyframe lidar sweep, refused when the scale is not positive | `labels.py`, `geometry.py:fit_scale_shift` |
| Masked BCE at pixels with a lidar return | `target` and `mask` per entry. A failed fit gives an empty mask, so the image adds nothing to the loss | `dataset.py` |
| Sparsification curve needs the true error per pixel | `abs_rel` is returned with every sample | `dataset.py` |
| Raw temporal variance as a baseline | A cached channel, so it costs nothing | `cache/build.py` |
| Ablations without radar, or on features alone | `HeadInputs(features=..., radar=...)` switches channel groups off at load time | `dataset.py` |
| Report on the official val split | `[splits]` holds the devkit's 700/150 split. The day, night, and rain groups come from BEVCar, under What differs | `configs/nuscenes-trainval.toml` |
| Depth range 1 m to 80 m | `points.min_depth = 1.0`, `points.max_depth = 80.0` | `cache/config.py` |

## What differs

| Proposal says | Pipeline does | What to do |
|---|---|---|
| Flip-augmentation disagreement as a baseline | Nothing in the cache stores a pass on the flipped image | Action 1 in [training-next-actions.md](training-next-actions.md): compute it for the val entries into a side file, with no rebuild. |
| Weather split from the BEVCar release of the nuScenes val split | `condition` is `night` when the scene description contains "night"; `weather` is `rain` when it contains "rain" | Action 2 in [training-next-actions.md](training-next-actions.md): report on BEVCar's scene lists at evaluation time, with no rebuild. |
| The frame selector scored per frame, on the same axes as the head | Not built | Under Implementations in [training-next-actions.md](training-next-actions.md), once the head works. |

## Accepted deviations

| Proposal says | Pipeline does | Why we accept it |
|---|---|---|
| MC dropout as a baseline, where the checkpoint permits | No MC dropout baseline | The DA-V2 encoder is built with every dropout rate at 0 (`depth_anything_v2/dinov2.py`), and the DPT head has no dropout, so MC passes give identical outputs. Forcing dropout on would measure damage to a model never trained with it, at about 10 passes per image. The proposal made this baseline conditional on the checkpoint, and flip disagreement (action 1) covers its role. The report says so in one sentence. |
| One confidence channel at image resolution | Everything is on the 264 x 148 grid at `patch_multiple = 4`. One cell covers about 6 x 6 pixels. The head predicts on the grid | Each label is the AbsRel at one lidar point, read from the full-resolution depth and then placed in its cell; the nearest point wins a cell. Every method is scored on the same labeled cells, so the grid costs each one equally. Full resolution would sharpen object boundaries in figures at the cost of a rebuild. Figures upsample the map to the image size, and the report states the grid size. |
| Under one percent of pixels carry a label | About 9% of the 264 x 148 grid cells carry a label, about 3,600 per image. The 26% measured earlier was on the 132 x 74 grid, where about 30% of the points share a cell and drop out | Both are true. Measured on 300 mini images, a camera image has about 3,650 lidar points, 0.25% of its pixels, and nearly every point gets its own cell. The report gives the cell number, which is what the head trains on, and the pixel number for comparison with the proposal. Lidar leaves the sky and tall structures unlabeled, so the head is neither trained nor scored there. Label counts per weather group are part of action 2. |
| Radar residual: predicted depth against radar range | The residual uses a scale-and-shift fit to the same radar points it measures (`RadarConfig.align = "radar"`) | A deployed car has radar but no lidar, so the fit comes from radar. The fit has two numbers for a median of 94 points, so it absorbs only an error the whole image shares, and each point keeps its disagreement with the rest of the image. On mini, the radar fit lowers the median residual only from 0.28 (lidar fit) to 0.25. Both fits are in every entry's meta, and `RadarConfig(align="lidar")` switches with one line. The temporal channel uses the lidar fit (`[temporal] align = "lidar"`), so the report does not claim the inputs are free of lidar. What the radar channel adds is action 3. |
| Confidence maps broken out by textureless surfaces and object edges, not only by weather | Results are broken out by weather group only | A good analysis, but not a near-term goal. It needs no rebuild if it comes back: edges can come from jumps in the cached disparity, and textureless regions from the image gradient. |

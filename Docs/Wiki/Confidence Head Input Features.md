# Confidence Head Input Features

This page is for team members who train or score the [[Per-Pixel Depth Confidence Head]]. It describes the five inputs that `ConfidenceDataset` stacks into `x`, one card per input, and the label the head learns. [[Confidence Head Input Cache]] covers how the cache stores them.

Everything comes from two sources: one pass of the frozen Depth Anything V2 (DA2) ViT-B, and the nuScenes camera images, lidar, radar, and poses. The checkpoint is the stock relative model, `depth_anything_v2_vitb.pth`. It predicts disparity (inverse depth) up to an unknown scale and shift, with no units.

## The Input Tensor

`x` has C + 4 channels on the cache grid: 264 x 148 for the trainval cache, where one cell covers about 6 x 6 image pixels. `dataset.py:ConfidenceDataset.__getitem__` stacks them in this order:

| Channels | Name | Card |
|---|---|---|
| 0 to 63 | `feature_0` to `feature_63` | Decoder features |
| 64 | `depth` | Depth |
| 65 | `temporal_relstd` | Temporal variance |
| 66 | `radar_residual` | Radar residual |
| 67 | `radar_valid` | Radar validity |

`ConfidenceHead` standardizes each channel with its mean and standard deviation over a sample of train images. The raw scales on the cards below therefore do not matter to training. `HeadInputs(features=False)`, `depth=False`, `temporal=False`, and `radar=False` switch a group off for the ablations.

## The Scale-and-Shift Fits

DA2 gives no depth in meters. Three parts of the pipeline need one, and each makes it with a least-squares fit of two numbers per image, a scale and a shift, from the disparity to the inverse depth of a set of points. In the code, "metric depth" and "metric fit" always mean the disparity after such a fit.

| Needs a depth in meters | Points of the fit | When the fit fails |
|---|---|---|
| The labels | The keyframe lidar sweep, always | The label mask is empty, and the image adds nothing to the loss |
| The radar residual | The radar points (default), or lidar by config | Both radar channels are all zero |
| The temporal variance, at build time | The keyframe lidar sweep (`[temporal] align = "lidar"`) | The temporal channel is all zero |

No part falls back to another fit. The depth input itself uses no fit.

## Decoder Features

| | |
|---|---|
| Channels | 64, `feature_0` to `feature_63` |
| Values | Raw activations, no units |
| Cached as | `features`, float16 |
| Computed | At build time, from the frozen DA2 pass |
| Code | `depth.py`, `cache/build.py:derive` |

**What it is.** The last decoder feature map of DA2 (the `output_conv1` tap): 64 channels taken just before the convolutions that turn features into disparity. It holds what the model has worked out about each location right before it commits to a depth. Because the encoder is a vision transformer, each location can carry context from the whole image. The head's bet is that these features reveal the kinds of region where DA2 tends to be wrong.

**How it is computed.** DA2 runs on the image resized to 924 x 518. The tap comes out at 8 times the patch grid (528 x 296), and the cache area-averages it to the grid.

**Reading the values.** No single channel has a fixed meaning. The head learns which combinations matter.

**When it is empty.** Never.

## Depth

| | |
|---|---|
| Channel | `depth` |
| Values | Disparity over its image mean: about 1 on average, larger for nearer surfaces |
| Cached as | `disparity_grid`, float32, the raw disparity |
| Computed | At load time, from the cached disparity |
| Code | `dataset.py:normalized_disparity` |

**What it is.** DA2's own prediction, as relative inverse depth. It gives the head the layout of the scene: what is near, what is far, and where the depth jumps at object edges.

**How it is computed.** The cache area-averages the model's disparity to the grid. At load time, `normalized_disparity` divides it by its mean over the image, so every image has the same scale. No fit is involved, and the head never sees meters.

**Reading the values.** A cell above 1 is nearer than the average of its image, and a cell below 1 is farther.

**When it is empty.** Never.

## Temporal Variance

| | |
|---|---|
| Channel | `temporal_relstd` |
| Values | Relative spread, 0 or more. 0.05 means the inverse depth varies by about 5% across frames. The median on v1.0-mini is 0.021 |
| Cached as | `temporal_relstd`, float16. `temporal_count`, the number of frames that see each cell, is cached too but is not an input |
| Computed | At build time: four extra DA2 passes per image |
| Fit | Lidar |
| Code | `cache/temporal.py` |

**What it is.** How much DA2's depth for the same 3D point disagrees across nearby frames, once the car's own motion is taken out. For a static scene with correct depth and correct poses, every frame gives the same point, so the value is 0 even while the car drives.

**How it is computed.** The neighbors are the camera frames at offsets -2, -1, +1, and +2 at 12 Hz, so about 0.17 s either way. For each cell:

1. Lift the cell center to 3D with the keyframe depth, made metric by the lidar fit.
2. Move the point into each neighbor camera with the two camera poses, and read the neighbor's disparity there.
3. Make the neighbor's disparity metric with its own fit to the moved points, dropping the worst 10% of cells, because each image has its own unknown scale.
4. Bring the sample back into the keyframe camera, and keep its inverse depth.
5. Take the standard deviation over the mean of these inverse depths across the frames that see the cell.

**Reading the values.** The value rises where the frames disagree:

- on moving objects, because the motion compensation assumes a static world;
- at occlusions, where a surface is visible in one frame and hidden in another;
- where the keyframe depth is wrong, because a wrong depth sends the point to the wrong place in the neighbor;
- where DA2 flickers, giving the same surface different depths from frame to frame.

The last two make it a signal of depth error. The raw value with no learning is also a baseline in the evaluation.

**When it is empty.** The value is 0 where fewer than two frames see the cell, which is 0.5% of cells on v1.0-mini, mostly at the image edges. The whole map is 0 when the keyframe has no lidar fit or no neighbor frames. `temporal_count` is not an input, so the head cannot tell these zeros apart from frames that agree.

## Radar Residual

| | |
|---|---|
| Channel | `radar_residual` |
| Values | \|predicted depth − radar depth\| / radar depth. The median per-image median on v1.0-mini is 0.25 |
| Cached as | Not cached. Computed from the cached radar points |
| Computed | At load time |
| Fit | Radar (`RadarConfig.align = "radar"`) |
| Code | `dataset.py:radar_channels` |

**What it is.** The disagreement between DA2's depth and the depth that radar measures, at each radar point. A large value is meant to warn the head that the depth there is suspect.

**How it is computed.** The points are the last 5 sweeps of all five radars, moved into the keyframe camera. That gives a median of 94 points per image on v1.0-mini. Each point carries DA2's disparity at its pixel, read from the model output before it shrinks to the grid. The radar fit turns that disparity into a depth, and the residual compares it with the point's own depth. Each point then fills a block of 3 x 3 cells (`RadarConfig.radius = 1`), and the nearest point wins a cell. Every other cell is 0.

**Reading the values.** The fit is made to the same points that the residual measures, but it has only two numbers for a median of 94 points. It absorbs an error that the whole image shares, such as a wrong overall scale, and each point keeps its own disagreement with the rest of the image. On v1.0-mini, the radar fit lowers the median residual only from 0.28 (under the lidar fit) to 0.25. The labels also come after a per-image fit, so the residual and the labels both ignore errors that the whole image shares. On v1.0-mini the residual is about five times the depth error that lidar measures, even under the lidar fit. A likely cause is that nuScenes radar reports no elevation, so a point projects to a pixel that may show another surface. [Action 3](../training-next-actions.md) measures what this channel adds.

**When it is empty.** The value is 0 away from radar points. The whole map is 0 when the image has no radar fit, which happens for 13.5% of the v1.0-mini images.

## Radar Validity

| | |
|---|---|
| Channel | `radar_valid` |
| Values | 1 where `radar_residual` holds a measurement, 0 elsewhere |
| Cached as | Not cached |
| Computed | At load time, with the residual |
| Code | `dataset.py:radar_channels` |

**What it is.** The mask of the cells where the radar residual is a measurement.

**Why it exists.** A residual of 0 can mean that radar agrees with the depth, or that no radar point is near the cell. The mask tells the head which zeros are measurements.

**What it looks like.** nuScenes radar reports no elevation, so every point projects as if it sat at the mounting height of its sensor. The mask is a thin, broken band of cells near the horizon line of each image.

**When it is empty.** All 0 when the image has no radar fit.

## What the Head Learns From Them

The head outputs one logit per cell, `p_error`: the probability that DA2's depth in the cell is wrong by more than `tau` = 10%. The label comes from lidar, never from the inputs above:

- `abs_rel`: \|predicted depth − lidar depth\| / lidar depth at each lidar point, with the predicted depth from the lidar fit and from DA2's disparity at the point's pixel, read before the grid. Each point goes to its cell, and the nearest point wins a cell.
- `target`: 1 where `abs_rel` > `tau`, 0 elsewhere.
- `mask`: 1 where a cell holds a lidar point. That is about 9% of the 264 x 148 cells, about 3,600 per image.

The loss is a binary cross-entropy over the masked cells, with the positive cells weighted by the ratio of negatives to positives (`head.py:masked_bce`). The evaluation sorts the labeled cells by `p_error` and uses `abs_rel` for the sparsification curves.

## Related

- [[Confidence Head Input Cache]]: how the cache stores these arrays, and the v1.0-mini measurements.
- [[Per-Pixel Depth Confidence Head]]: the method these inputs feed.
- [[CS566 Confidence Head Training Run 1]]: the plan for the first training run.

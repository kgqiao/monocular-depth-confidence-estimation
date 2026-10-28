# Training Next Actions

Work that comes out of the comparison in [proposal-vs-pipeline.md](proposal-vs-pipeline.md). Each action names the deviation it closes.

## 1. Establish the Flip-Augmentation Baseline

Form B lists flip-augmentation disagreement as a cheap baseline, and Poggi et al. (CVPR 2020) report it as "Post". The cache holds no pass on the flipped image, and the full trainval cache is already built.

The baseline is scored on the val split only, so it needs no rebuild:

- Run DA-V2 on the horizontally flipped keyframe of each val entry (about 6,000 keyframes × 6 cameras ≈ 36,000 images), flip the output back, and resize it to the entry's grid.
- Align the flipped disparity to the keyframe disparity with a scale-and-shift fit that uses no lidar, then take the relative difference per cell as the confidence score.
- Write one grid per entry key to a side file next to the cache. The cache and its fingerprint stay as they are.
- Read the side file in the evaluation as one more baseline curve.

Done when the evaluation reports AUSE and AURG for the flip baseline on the val split.

## 2. Report on the BEVCar Weather Split

Form B asks for day, night, and rain results on the BEVCar release of the nuScenes val split. The cache tags each scene by keywords in its description instead (`nuscenes.py:condition_of`, `weather_of`). A scene can be both night and rain, and the substring "rain" also matches words such as "train" and "terrain".

BEVCar's [`custom_nuscenes_splits.py`](https://github.com/robot-learning-freiburg/BEVCar) lists all 150 val scenes in three disjoint groups: 111 day, 24 rain, and 15 night (4,449, 968, and 602 keyframes in the paper, arXiv 2403.11761). The README puts rainy night scenes in night; the code comment says rain holds both, but the lists do not overlap. Every index row holds its scene name, so the lists apply at evaluation time with no rebuild.

- Copy the three scene lists into the evaluation, with the source and a citation of Schramm et al., IROS 2024.
- Report every number on day, rain, and night from those lists, with the scene count next to each.
- On CARC, compare the cache's own tags with the BEVCar lists. Note where the keywords disagree, and state in the report which rule BEVCar follows for rainy night scenes.
- Write a label count table per BEVCar group: scenes, keyframes, camera images, labeled cells per image, and the share of labeled cells with AbsRel above `tau`. Form B and the TA ask for it, because wet roads return fewer lidar points, so rain may have the fewest labels where depth is most suspect.

The cache's own tags stay as they are.

Done when the evaluation reports AUSE and AURG per BEVCar group, the label count table is with the results, and the comparison of the two tag sets is in the worklog.

## 3. Measure What the Radar Channel Adds

The radar residual compares the DA-V2 depth with radar range after a scale-and-shift fit to the same radar points. On the 2,424 mini images, the residual looks more like radar error than depth error:

| Measure | Value |
|---|---|
| Radar points per image, 5 sweeps | mean 150, median 94 |
| Images with no radar fit, so all-zero radar channels | 13.5% |
| Median per-image median radar residual, radar fit | 0.25 |
| Same, lidar fit | 0.28 |
| Median per-image median AbsRel against lidar, day | 0.05 |

The residual is about five times the depth error that lidar measures, even under the lidar fit. A likely cause is that nuScenes radar has no elevation, so a radar point projects to a pixel that may show another surface. The radar fit is also plain least squares (`geometry.py:fit_scale_shift`), so a few bad points pull it.

- Train and score the head three ways: radar off (`HeadInputs(radar=False)`), the radar fit (the default), and the lidar fit (`RadarConfig(align="lidar")`).
- Try a trimmed radar fit, as the temporal fit already uses, computed at load time from the cached radar points. Check whether it lowers the residual or the share of images with no fit.
- Report the radar residual next to the lidar AbsRel, so the noise level is on record.

None of this needs a rebuild: every radar point is cached with the model disparity at its pixel, and the radar channels are computed at load time.

Done when the evaluation reports AUSE and AURG for the three variants, and the report says whether the radar channel helps.

## Implementations

Uses of the head that demonstrate it in practical situations. They start once the head works.

### Frame Selector

Form B keeps the frame selector of the original plan as a secondary contribution: a hand-designed rule that ranks frames by depth variance and keeps the best 10 to 15%. The head gives a learned version of the same idea.

- Rank whole camera images by their mean `p_error`, drop the worst N%, and recompute the depth error on the images that remain. This is the sparsification curve with the image as the unit.
- Score the hand-designed rule the same way, with images ranked by their mean raw temporal variance.
- Report both on the same axes, so the selector gets a number and the head is compared with it.

The cache needs no rebuild: both rankings come from arrays that every entry already holds.

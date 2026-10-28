---
id: doc-1r2ch6
type: topic
kind: concept
description: Where the confhead code at ff2bc31 differs from the Run 1 plan, Form B, the pre-proposal, and the dataset strategy, and the open questions the team must settle before training.
tags: [machine-learning/cs566-deeplearning/cs566-project, machine-learning/cs566-deeplearning/cs566-project/kqiao-cs566-project-repo]
aliases: [confhead deviations, CS566 open questions]
created: 2026-10-19T17:34:12
updated: 2026-10-19T17:35:36
refreshed: 2026-10-19T17:35:36
status: stable
sources: ["[[kqiao-cs566-project-repo @ ff2bc31]]", "[[CS566 Confidence Head Training Run 1 (plan, 2026-10-12)]]", "[[CS566 Brainstorm - Project Scope Visual (Form B)]]", "[[CS566 Project Pre-proposal]]", "[[CS566 Dataset Strategy (plan, 2026-10-12)]]", "[[CS566 Project Pre-preposal Feedback]]"]
---

> [!concept] Concept
> 6 sources · refreshed 2026-10-19 · #machine-learning/cs566-deeplearning/cs566-project · #machine-learning/cs566-deeplearning/cs566-project/kqiao-cs566-project-repo

## Definition

This page lists each place where the code of the CS566 project departs from the plans, and the questions the team must settle. The code is the [[confhead library]] at `ff2bc31` on branch `work/nathan-aday` of [[kqiao-cs566-project-repo]]. The plans are [[CS566 Confidence Head Training Run 1]] (Run 1), Form B ([[CS566 Brainstorm - Project Scope Visual (Form B)]]), the pre-proposal ([[CS566 Project Pre-proposal]]), and [[CS566 Dataset Strategy]]. The plans do not agree with each other on every point, so each row names the plan it compares with. Update this page when the code or a plan changes.

## Explanation

### Deviations

| # | Subject | The plan says | The code does | Status |
|---|---|---|---|---|
| D1 | Depth model | Run 1: the metric checkpoint `depth_anything_v2_metric_vkitti_vitb.pth`, in meters, with no per-image fit. The pre-proposal measured errors after a per-image fit to lidar (p. 1). | The relative ViT-B checkpoint. Each entry keeps a least-squares fit of disparity to inverse depth against the keyframe lidar and one against the radar. The labels use the lidar fit. | Follows the pre-proposal, not Run 1. The TA's leakage warning applies ([[CS566 Project Pre-preposal Feedback]]). See Q2. |
| D2 | Feature storage | Run 1: the model runs inside the training loop; no feature store, because 128 x 296 x 528 per image would need about 8 TB. The pre-proposal (p. 2, §2.1) and Form B cache the features. | The cache stores the features on a coarser grid: 5.4 MB per image at the default grid, 13 GB for mini. | Follows the pre-proposal and Form B. See Q3. |
| D3 | Feature map | Run 1: `path_1`, 128 channels at 296 x 528. Form B: C of about 64 to 128. | Default tap `output_conv1`, 64 channels, area-averaged to 264 x 148 (the patch grid times 4). `path_1` is the tap `refinenet1`, a setting. | Within Form B; coarser than Run 1. See Q3. |
| D4 | Temporal channel | Run 1: none until Run 2. Form B: per-pixel standard deviation of depth over plus or minus 2 frames, with ego-motion compensation. | Built: [[Ego-Motion-Compensated Temporal Spread]], std / mean of inverse depth per grid cell, at the cost of 4 more model passes per image. | Follows Form B, with a relative statistic. |
| D5 | Depth channel | Run 1: D / 80, in meters. | Disparity over its image mean. | Follows from D1. |
| D6 | Radar residual | Run 1: \|D - z\| / z clipped to [0, 1], spread over a radius of 2 cells at 296 x 528, from the keyframe radar. | \|pred - z\| / z with the depth from the radar fit, not clipped, over a radius of 1 grid cell, from up to 5 sweeps per radar (about 150 points per image against 31). Empty when the radar fit fails: 13.5% of mini images. | See Q4. |
| D7 | Where a label sits | Run 1: sample the logits and D at the lidar pixel coordinates (bilinear), one label per point. | The lidar AbsRel is put on the grid; the nearest point wins a cell; one label per cell. | See Q5. |
| D8 | CAM_BACK ego body | Run 1, step 3: drop the CAM_BACK lidar points that the trunk hides (3.2% of CAM_BACK lidar pixels; [[Lidar and Radar Projection into Camera Images]]). | Not masked. | Open: Q1. |
| D9 | Occlusion bleed-through | About 0.35% of lidar pixels are points behind a nearer object ([[Lidar and Radar Projection into Camera Images]]). | Not removed, from the labels or from the temporal spread. | Open: Q6. |
| D10 | Data | Run 1 and the dataset strategy: full nuScenes, with train, dev, and report splits. | Built and checked on v1.0-mini only, with the official mini split (8 train scenes, 2 val); no dev split and no full-nuScenes config yet. `utilities/dataset/` holds a CARC job that downloads the full nuScenes. | Next step. |
| D11 | Weather tag | Dataset strategy: decide first how to count "Night, after rain". | Rain when the description holds "rain", so scene-1094 is night and rain (`confhead/src/confhead/nuscenes.py:104`). | Open: Q7. |
| D12 | Label gate | Run 1: compute the positive-label rate on train; act when it is outside 0.05 to 0.70. | On mini, 37.2% at tau 0.10. | Inside the gate, but measured with the relative model and the lidar fit (D1). |
| D13 | Images with no fit | Not covered. | 4 night images of scene-1077 have no lidar fit, so no labels and no temporal spread. | Open: Q8. |
| D14 | Choice of checkpoint | Run 1, step 6: compute AUSE on dev after each epoch, and keep the checkpoint with the lowest dev AUSE. | `confhead/notebooks/train-head.ipynb` keeps the epoch with the lowest masked loss on dev, the loss it trains on, and computes no scores. On mini, the two rules disagree: dev AUSE was lowest at epoch 1 (the notebook at `c0e979a`), and the dev loss at epoch 8. | Decided: the eval team owns AUSE and the other scores, and scores the checkpoint that the dev loss chose. |

### Where the code follows the plans

- The input is Form B's `[C+4, h, w]`: features, depth, temporal spread, radar residual, radar validity ([[Per-Pixel Depth Confidence Head]]).
- The label is AbsRel > 0.10 against lidar, masked to the cells with a lidar return; tau is a setting at load time.
- The model is frozen.
- The projection conventions: depth along the optical axis, 1 m to 80 m, the nearest point wins, the devkit radar filter.
- Every entry carries its condition (day, night) and weather (clear, rain), so results split per condition.

### Open questions

1. **CAM_BACK ego mask (D8).** Mask a fixed image region per camera (CAM_BACK rows 820 and below on mini), or drop points by a depth test? Decide before training: these labels are road that the camera cannot see.
2. **Relative or metric depth (D1, D5, D12).** Keep the relative model with the lidar fit (the standard evaluation, but it uses lidar the car does not have, and it can leak into the labels), move to the metric checkpoint of Run 1, or use the radar fit at inference?
3. **Store the features, or run the model in the loop (D2, D3)?** A synthesis from the mini sizes: the 204,894 train and val camera images would take about 1.1 TB at the default grid, or about 290 GB at the 132 x 74 grid. The cache trades disk for a model pass per training step, and a grid coarser than Run 1's 296 x 528.
4. **Radar residual (D6).** Clip it or not; which radius; how many sweeps; which fit; and what the channel holds when the radar fit fails.
5. **Labels per point or per grid cell (D7)?** Per point keeps every lidar return; per cell matches the head's output grid.
6. **Occlusion (D9).** Remove bleed-through points from the labels? Leave occluded neighbors out of the temporal spread?
7. **Weather tag (D11).** Does "Night, after rain" count as rain? Compare with the BEVCar val weather split ([[CS566 Dataset Strategy]]).
8. **Images with no fit (D13).** Drop them, or label every pixel as wrong? They are real failures of the model at night.
9. **Which error figure to report.** The pre-proposal quotes about 10% relative error by day and 20% at night after the fit (p. 1). The cache gives a median per-image median AbsRel of 0.052 by day and 0.079 at night on labeled cells ([[Confidence Head Input Cache]]). The statistics differ; the report should name one.

## Related

- [[CS566 Confidence Head Training Run 1]]: the Run 1 plan.
- [[Per-Pixel Depth Confidence Head]]: the method, from the pre-proposal and Form B.
- [[CS566 Dataset Strategy]]: the data and the splits.
- [[confhead library]], [[Confidence Head Input Cache]], [[Ego-Motion-Compensated Temporal Spread]]: the code.
- [[Lidar and Radar Projection into Camera Images]]: the label pitfalls.

## Sources

- [[kqiao-cs566-project-repo @ ff2bc31]]
- [[CS566 Confidence Head Training Run 1 (plan, 2026-10-12)]]
- [[CS566 Brainstorm - Project Scope Visual (Form B)]]
- [[CS566 Project Pre-proposal]]
- [[CS566 Dataset Strategy (plan, 2026-10-12)]]
- [[CS566 Project Pre-preposal Feedback]]

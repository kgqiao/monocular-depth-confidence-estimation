---
id: doc-h91hyg
type: topic
kind: concept
description: "The plan for the first CS566 confidence-head training run: full nuScenes, frozen metric Depth Anything V2 ViT-B, radar inputs, lidar labels, steps, compute, and done criteria."
tags: [machine-learning/cs566-deeplearning/cs566-project]
aliases: [Training Run 1, Run 1]
created: 2026-10-12T15:34:33
updated: 2026-10-19T17:35:36
refreshed: 2026-10-19T17:35:36
status: stable
sources: ["[[CS566 Confidence Head Training Run 1 (plan, 2026-10-12)]]", "[[CS566 Project Pre-preposal Feedback]]", "[[Depth Anything V2]]", "[[Lidar and Radar Projection into Camera Images]]"]
---

> [!concept] Concept
> 4 sources · refreshed 2026-10-19 · #machine-learning/cs566-deeplearning/cs566-project

## Definition

The plan for the first training run of the [[Per-Pixel Depth Confidence Head]], written by the Datasets and Training role on 2026-10-12 ([[CS566 Confidence Head Training Run 1 (plan, 2026-10-12)]]). It says what the run does and how. The theory is in [[Per-Pixel Depth Confidence Head]]; the data and the splits are in [[CS566 Dataset Strategy]].

## Explanation

### Goal

Train the head once on full nuScenes, on a frozen [[Depth Anything V2]], and write a confidence map for every camera image of the report split (official val). Run 1 answers one question: can the frozen model's features, with radar, predict where its own depth is wrong? Run 1 is done when the items under "Done when" are true. Good numbers are not a condition ([[CS566 Confidence Head Training Run 1 (plan, 2026-10-12)]], Goal).

### The fusion pipeline

At inference the head reads the camera and the radar. Lidar is used only in training, to make the labels.

```
camera image ──► Depth Anything V2 metric ViT-B (frozen) ──► metric depth D
                                         └──────────────► decoder features F (128 ch)
radar points ──► project into the image ──► residual |D − z_radar| / z_radar, validity mask
[F, D, residual, mask] ──► head (4 conv layers) ──► p_error per pixel
lidar points ──► project into the image ──► label y = 1 where |D − z_lidar| / z_lidar > 0.10   (training only)
```

### Decisions

From [[CS566 Confidence Head Training Run 1 (plan, 2026-10-12)]], Decisions for Run 1:

1. **Depth model: the metric outdoor checkpoint, `depth_anything_v2_metric_vkitti_vitb.pth`.** This answers the TA's label risk ([[CS566 Project Pre-preposal Feedback]]). The checkpoint outputs meters (max 80 m), so no per-image fit to lidar is needed. Its fine-tuning used synthetic Virtual KITTI 2, not nuScenes, so no nuScenes lidar leaks into the labels. The relative checkpoint and the per-image lidar fit used so far are dropped: that fit needs lidar, which the car does not have at inference.
2. **Label gate.** Before training, compute the positive-label rate (share of lidar pixels with AbsRel > 0.10) on train. If it is above 0.70 or below 0.05, fit one scale factor per camera on the train split only (median of z_lidar / D) and apply it everywhere. The constant is known at inference and never reads report-split lidar.
3. **Inputs: features, depth, radar residual, radar mask (131 channels).** Temporal variance waits for Run 2. Run 1 tests the pre-proposal's open question, whether confidence can be read from a frozen model's features, without the 4 extra depth passes per image and the ego-motion reprojection that variance needs.
4. **Depth Anything runs inside the training loop, with no gradient.** The decoder features are 128 x 296 x 528 per image, about 40 MB in fp16; 205k images would need about 8 TB. A frozen model in the loop needs no feature store. Only the sparse lidar and radar points go to disk.
5. **Data:** all 6 cameras and all keyframes of the train, dev, and report splits.

### Steps

The code goes in the `confhead` package of [[kqiao-cs566-project-repo]], next to the loader and the projection that exist ([[CS566 Confidence Head Training Run 1 (plan, 2026-10-12)]], Steps). When the plan was written, that package was in `pipeline/`; it is now the [[confhead library]] in `confhead/`. The cache that the library builds departs from steps 1 to 5 in several ways: see [[CS566 Implementation Deviations and Open Questions]].

1. **Port the metric model.** The vendored `dpt.py` is the relative model: its output ends in ReLU. Add the metric output from upstream `metric_depth/`: a Sigmoid times `max_depth` (80). Make the forward pass also return `path_1`, the 128-channel fused decoder map (296 x 528 for a 518 x 924 input).
2. **Index.** One table per split: each keyframe camera image with its path, scene, condition, intrinsics, and pose, built with the numpy loader.
3. **Point cache (one CPU pass).** For each camera image, store the projected lidar and radar points (u, v, z) with the existing conventions: depth along the camera axis, 1 to 80 m, nearest point wins, devkit radar filter. Drop the CAM_BACK lidar points on the ego car's trunk ([[Lidar and Radar Projection into Camera Images]]). Estimate: about 43 KB per image, about 10 GB in all.
4. **Label statistics.** Run the frozen model once over train, dev, and report. Write the counts table of [[CS566 Dataset Strategy]] and the positive-label rate per split and condition. Apply the label gate.
5. **Training loop, per batch:**
   1. Load the JPEG, resize to 518 x 924, ImageNet-normalize.
   2. Run the frozen model in fp16 under `torch.no_grad()`: depth D (518 x 924) and features F (128 x 296 x 528).
   3. Resample D to 296 x 528.
   4. Radar channels at 296 x 528: at each radar point, residual = |D − z_radar| / z_radar, clipped to [0, 1], spread over a radius of 2 cells; mask = 1 there, 0 elsewhere.
   5. Stack [F, D / 80, residual, mask]: 131 channels.
   6. Head: conv 3x3 131→64, ReLU; conv 3x3 64→64, ReLU; conv 3x3 64→32, ReLU; conv 1x1 32→1. About 0.13M parameters; logits at 296 x 528.
   7. Sample the logits and D at the lidar pixel coordinates with `grid_sample` (bilinear). Label y = 1 where |D − z| / z > 0.10.
   8. Loss: binary cross-entropy with logits over the lidar pixels only, `pos_weight` = negatives / positives on train.
6. **Optimization:** AdamW, learning rate 1e-3, weight decay 1e-4, cosine schedule, batch 8 images, 8 epochs, fixed seed. After each epoch, compute AUSE on dev, and keep the checkpoint with the lowest dev AUSE.
7. **Prediction on the report split.** For every image, write p_error at 296 x 528 (fp16, about 0.3 MB; about 11 GB for 36,114 images). Also write one evaluation file for the Evaluation team: for each lidar pixel, the image, scene, condition, u, v, z, D, and p_error. The random and oracle baselines come from that file alone.

### Compute

CARC, one GPU (A100 or A40). The time per epoch is not known yet: measure images per second on the first 1,000 train images and record it. If an epoch takes more than 2 hours, train each epoch on a random half of the train keyframes ([[CS566 Confidence Head Training Run 1 (plan, 2026-10-12)]], Compute).

### Done when

- The metric model port reproduces the upstream metric output on one image.
- The counts table and the positive-label rates are in `results/`.
- Training finishes; the config, git commit, loss curve, and dev AUSE per epoch are saved with the checkpoint.
- Every report-split image has a confidence map, and the evaluation file exists.
- A short note gives AUSE on the report split for the head, random removal, and the oracle: all conditions, day, night, and rain, with the scene count next to each number.

### Run 2 (not part of Run 1)

Add the temporal variance channel and the raw-variance baseline. Run the ablations without radar and with features only. Add one downstream consumer to the main results, as the TA asked, for example confidence-weighted depth fusion ([[CS566 Project Pre-preposal Feedback]]).

## Related

- [[CS566 Dataset Strategy]]: the data, splits, and counts this run uses.
- [[Per-Pixel Depth Confidence Head]]: the method and its theory.
- [[Depth Anything V2]]: the frozen backbone.
- [[Lidar and Radar Projection into Camera Images]]: the projection code the point cache reuses.
- [[Train the confidence head on nuScenes trainval]]: the thread stub this plan specifies.
- [[kqiao-cs566-project-repo]]: where the code goes.
- [[confhead library]]: the library the code is in.
- [[CS566 Implementation Deviations and Open Questions]]: where the built code differs from this plan.

## Sources

- [[CS566 Confidence Head Training Run 1 (plan, 2026-10-12)]]: the plan, with the checked metric-checkpoint facts.
- [[CS566 Project Pre-preposal Feedback]]: the label, leakage, and downstream-consumer points the plan answers.
- [[Depth Anything V2]]: the relative model and lidar fit used so far.
- [[Lidar and Radar Projection into Camera Images]]: the projection conventions and the CAM_BACK trunk.

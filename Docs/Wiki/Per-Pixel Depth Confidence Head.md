---
id: con-e5wghd
type: topic
kind: concept
description: "The CS566 project's planned method: a small CNN on top of a frozen Depth Anything V2 that predicts, per pixel, the probability that the depth error exceeds a threshold, trained on lidar-projected error labels and scored with sparsification curves."
tags: [machine-learning/cs566-deeplearning/cs566-project]
aliases: [confidence head, learned confidence head, learned confidence estimator, Predicting where Depth Anything is wrong]
created: 2026-09-28T00:00:00
updated: 2026-10-19T17:35:46
refreshed: 2026-10-19T17:35:46
status: stable
sources: ["[[CS566 Project Pre-proposal]]", "[[CS566 Brainstorm - Project Scope Visual (Form B)]]", "[[Load nuScenes mini and project lidar and radar into the cameras · Spec]]", "[[Load nuScenes mini and project lidar and radar into the cameras · Verification 1]]", "[[Load nuScenes mini and project lidar and radar into the cameras · Verification 4]]", "[[CS566 Project Pre-preposal Feedback]]"]
---

> [!concept] Concept
> 6 sources · refreshed 2026-10-19 · #machine-learning/cs566-deeplearning/cs566-project

## Definition

The per-pixel depth confidence head is the method of the CS566 group project. It is a small convolutional network on top of a frozen [[Depth Anything V2]]. For each pixel, it outputs the probability that the depth there is wrong by more than a threshold ([[CS566 Project Pre-proposal]], p. 1). The output is a confidence map, not a better depth map ([[CS566 Project Pre-proposal]], p. 2).

This page describes a plan. The pre-proposal proposes the method, and the team has not trained it yet. [[CS566 Brainstorm - Project Scope Visual (Form B)]] is the planning document that led to the pre-proposal. Where the two differ, this page says so. The first proof-of-concept step, the lidar and radar projections, is built and measured: see [[Lidar and Radar Projection into Camera Images]]. The cache of the head's inputs is built on nuScenes v1.0-mini: see [[Confidence Head Input Cache]]. Where the code differs from this plan is in [[CS566 Implementation Deviations and Open Questions]].

## Explanation

### The problem

- Depth Anything and similar models supply depth to systems with no depth sensor, in driving, robotics, AR, and 3D reconstruction, usually as a drop-in component. Every pixel gets a depth value and no measure of reliability: "an estimate that is wrong by thirty percent looks exactly like one that is right" ([[CS566 Project Pre-proposal]], p. 1).
- Depth degrades at night, through a rain-covered lens, and on object boundaries ([[CS566 Project Pre-proposal]], p. 1). Night, rain on the lens, spray from other vehicles, and low sun are when camera depth is least reliable and when wrong depth costs the most ([[CS566 Project Pre-proposal]], p. 1, §1.1).
- In the team's own runs on [[nuScenes]], Depth Anything V2 needs a per-image scale and shift fitted to lidar before its output is in meters. After that fit, the relative error is about ten percent by day and twenty percent at night ([[CS566 Project Pre-proposal]], p. 1).
- A confidence map lets a downstream consumer weight the depth, discard it, or fall back to another sensor ([[CS566 Project Pre-proposal]], p. 1).
- Depth Anything is a large supervised model that people use frozen and cannot retrain. Whether confidence can be read from its features without retraining is an open question ([[CS566 Project Pre-proposal]], p. 2, §1.2).
- [[HyDRa]] and [[LiCROcc]] use radar to improve the depth or the scene itself. This project uses radar to predict where the depth is wrong and leaves the depth model untouched ([[CS566 Project Pre-proposal]], p. 2).
- The method is not specific to driving: any system that consumes monocular depth has the same problem ([[CS566 Project Pre-proposal]], p. 1).

### The method

**Frozen backbone.** Depth Anything V2 gets no gradient updates. Each image passes through it once, and the predicted depth and decoder features are cached for training the head ([[CS566 Project Pre-proposal]], p. 2, §2.1).

**Head.** About four convolutional layers and single-digit millions of parameters. It is the only component whose weights change ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § What the confidence head is; [[CS566 Project Pre-proposal]], p. 1).

**Inputs.** One tensor per camera image, shape `[B, C+4, H, W]` ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § What the confidence head is; the same channels are in [[CS566 Project Pre-proposal]], p. 2, §2.2):

| Channels | Count | Source |
|---|---|---|
| Depth Anything V2 decoder features | C (about 64 to 128) | One forward pass of the frozen model |
| Predicted depth | 1 | The same pass |
| Temporal depth variance | 1 | Depth on neighboring sweeps, reprojected into this frame, per-pixel standard deviation |
| Radar residual | 1 | \|predicted depth - radar range\| at pixels where a radar point projects |
| Radar validity mask | 1 | 1 where a radar point projects within a few pixels, 0 elsewhere |

The Form B pipeline uses the 12 Hz sweeps, plus or minus 2 frames, with ego-motion compensation for the variance channel ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § The pipeline). The project pipeline measures radar as depth along the camera axis rather than range, so the residual compares like with like ([[Load nuScenes mini and project lidar and radar into the cameras · Spec]], Decisions).

**Output.** One channel at image resolution: the probability that the depth error at the pixel exceeds a fixed threshold ([[CS566 Project Pre-proposal]], p. 2, §2.2). Form B starts with AbsRel > 0.10 and says to report the choice. It notes that regressing the error value also works, but the threshold version is easier to calibrate and explain ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § What the confidence head is).

**Labels.** Project the lidar sweep into the image. At each pixel with a lidar return, the true depth gives the true error of the Depth Anything prediction. No hand annotation is needed ([[CS566 Project Pre-proposal]], p. 1). Lidar is present when the model is built and absent when it runs, so the head must predict an error it cannot measure ([[CS566 Project Pre-proposal]], p. 1).

**Loss.** Binary cross-entropy at pixels with a lidar return, masked everywhere else. Under one percent of pixels carry a label. The head trains on these scattered pixels and outputs a dense map at inference ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § What the confidence head is; [[CS566 Project Pre-proposal]], p. 1). Measured on nuScenes v1.0-mini with one keyframe sweep: 0.25% of pixels, about 3,600 per 1600 x 900 image, from 0.21% in CAM_FRONT to 0.31% in CAM_BACK, and about the same by day and at night ([[Lidar and Radar Projection into Camera Images]]).

### Training in three passes

[[CS566 Brainstorm - Project Scope Visual (Form B)]], § Training Path, plans three passes:

1. **Cache.** Run Depth Anything forward only. Write the depth map, decoder features, temporal variance, radar residual and mask, and lidar error map to disk for each image. The team's sandbox already does the depth part at about a minute per scene on an Apple GPU.
2. **Train.** Train the head on the cached tensors with masked BCE. Depth Anything is not in the graph, so training takes hours on one GPU, not days.
3. **Evaluate.** Run the head on held-out scenes to produce sparsification curves and split tables.

Form B notes that the team can use CARC GPU time. The pre-proposal keeps the caching step (p. 2, §2.1) but does not describe the passes.

### Baselines

The pre-proposal compares the head with random removal, raw temporal variance, flip-augmentation disagreement, and MC dropout where the checkpoint permits ([[CS566 Project Pre-proposal]], p. 2, §3.1). Form B adds the oracle to the list. It calls raw temporal variance the direct test of the original frame-selector assumption: if the head cannot beat it, that is a result ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § How it is scored).

### Evaluation

- The primary score is the [[Sparsification Curve]], with AUSE and AURG from Poggi et al., 2020 ([[CS566 Project Pre-proposal]], p. 2, §3.1).
- Every result is reported separately for day, night, and rain on the nuScenes validation split ([[CS566 Project Pre-proposal]], p. 2, §3). Form B names the split: the BEVCar release of the nuScenes val weather split, with the scene count next to each number ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § How it is scored).
- The team expects the largest margin over variance at night and in rain. If the head does not beat variance, the negative result goes in the same table. Ablations without the radar channels and on decoder features alone decide what stays in the model ([[CS566 Project Pre-proposal]], p. 2, §3.1).
- Form B also asks for label density per weather split, and for confidence maps broken out by textureless surfaces and object edges ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Things to keep in mind).
- Deliverables: confidence maps per condition, one table (AUSE and AURG per method and condition), one plot (sparsification curves), and a repository ([[CS566 Project Pre-proposal]], p. 2, §3.3).

### Datasets

The method needs four things in the same recording: lidar projected into the image (the only source of error labels), camera sequences (temporal variance needs neighboring frames), calibrated radar (two input channels), and condition labels (to split results by lighting and weather) ([[CS566 Project Pre-proposal]], p. 2, §4).

- **[[nuScenes]]** is the only public set with all four. It has 1,000 scenes of 20 seconds from Boston and Singapore, 2 Hz annotation with 40,000 keyframes, and unannotated sweeps between them for temporal variance. The official split is 700 training, 150 validation, and 150 test scenes ([[CS566 Project Pre-proposal]], p. 3, §4.1). It is for training and evaluation (p. 3, Table 1).
- **STF / DENSE** covers fog and snow that nuScenes lacks: about 12,000 on-road samples from more than 10,000 km in northern Europe, plus 1,500 from a fog chamber where visibility is set, so weather severity is a controlled variable ([[CS566 Project Pre-proposal]], p. 3, §4.2). It is the candidate for heavier weather (p. 3, Table 1).
- **Zenseact ZOD** is the backup. It has no radar, so it becomes usable only if the ablation shows radar was not carrying the head ([[CS566 Project Pre-proposal]], p. 3, §4.3).
- Form B rejects DAWN: it has no lidar, radar, or sequences, so the head cannot run on it ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Datasets).
- Form B treats transfer from nuScenes to another set as a domain shift: different lidar, intrinsics, camera count, and radar. The radar channels will not transfer without recomputation. It scopes transfer as a stretch goal with a fallback: if transfer fails, the team reports the weather claim on nuScenes and calls it mild ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Datasets).

### Stretch goals

**Frame selector.** The frame selector of the team's first plan ranks frames by depth variance, checks suspect regions against radar, and keeps the best 10 to 15 percent. Form B calls the selector the hand-designed version of the idea and the head the learned version. It scores the selector with the same curve, with the frame as the unit: rank frames by mean predicted confidence and drop the worst N percent. The selector drops frames. The head gives a dense map for every frame, so a consumer that needs output for every frame, such as occupancy, can use only the head ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Secondary contribution: the frame selector).

**Confidence-weighted occupancy.** An occupancy network weights its camera-to-voxel lifting by confidence, so unstable pixels contribute less ([[CS566 Project Pre-proposal]], pp. 1–2). The team plans to measure occupancy mIoU with and without the head ([[CS566 Project Pre-proposal]], p. 2, §3.2). Form B's Stage 2 uses a 200 x 200 x 16 grid with 17 classes, scored by mIoU for day, night, and rain against Occ3D labels. It trains a full occupancy network and carries the compute risk. Stage 1 stands on its own without it ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Stage 2).

### Risks and open questions

From [[CS566 Brainstorm - Project Scope Visual (Form B)]], § Things to keep in mind and § Next Steps After Pre-Proposal:

- **Missing labels in rain.** Wet asphalt is specular and returns fewer lidar points. Rain gives fewer labels where camera depth is most suspect.
- **Variance is not weather-specific.** Textureless surfaces and object edges also make depth unstable. Variance alone cannot tell weather from geometry, so the head must learn the difference from the decoder features.
- **Sparse radar.** Five radars return a few hundred points per keyframe, projected into six images, so the radar channels are mostly zeros. The ablation tests whether they help. Measured on mini: about 31 radar points per image, 0.002% of the pixels, and 61 of 2,424 images get none ([[Lidar and Radar Projection into Camera Images]]).
- **Mild weather in nuScenes.** Published occupancy work reports lidar-only mIoU dropping only a few points from clear day to rainy night. The team plans to find and cite the exact number, because it decides how hard to push the weather framing.
- **Open checks:** whether the BEVCar split is usable as published; which Poggi et al. baselines transfer to a supervised foundation model; whether anyone has published depth-uncertainty numbers for Depth Anything (existing work is mostly KITTI and Monodepth2); whether STF / DENSE ships sequences; access lead time for STF / DENSE and ZOD. The fraction of pixels with a lidar return is answered above. The decoder feature size and the cache size are answered too: the last decoder maps of ViT-B have 64 or 128 channels at 8 times the patch grid, and the cache takes 1.4 MB per camera image on a 132 x 74 grid, or 5.4 MB on 264 x 148 ([[Confidence Head Input Cache]]).
- **First experiment.** Plot per-pixel depth variance against actual lidar error, colored by weather tag. A weak correlation means the variance channel is dead weight.

Found by the projection thread ([[Load nuScenes mini and project lidar and radar into the cameras · Verification 1]], F3; [[Lidar and Radar Projection into Camera Images]]):

- **Labels the camera cannot see.** The lidar sits above the cameras. In CAM_BACK, 3.2% of lidar pixels are road behind the ego car's trunk, so the labels need an ego-body mask there. Elsewhere, about 0.35% of lidar pixels look like points behind a nearer object.
- **No labels on glass.** Glass facades return almost no lidar, so the head gets almost no supervision there.

### TA feedback on the pre-proposal

The TA calls the pre-proposal very well specified, with a clear baseline hierarchy and an honest negative-result plan. The TA names the labels as the main technical risk ([[CS566 Project Pre-preposal Feedback]]):

- **Metric labels.** The default Depth Anything V2 models output relative (affine-invariant) depth. To get metric labels, the team needs either the metric outdoor checkpoint or a per-frame scale and shift alignment.
- **Leakage into the labels.** An alignment fitted to the same lidar that the head is evaluated against can leak information into the labels. Synthesis, not the TA's claim: the measured errors under [[Depth Anything V2]] use a per-image fit to lidar, so the warning applies to the current pipeline.
- **Empty radar residual.** nuScenes radar is very sparse and has no reliable elevation, so the radar residual may be empty for most pixels.
- **Few adverse scenes.** Night and rain scenes are a small share of the validation split. Count the labeled pixels per condition before committing to per-condition claims.
- **A downstream consumer in the main results.** Move one downstream consumer from the stretch goals into the main results, even a simple one such as confidence-weighted depth fusion or occupancy. In the TA's words: "A confidence map is convincing when it changes a downstream decision, not only when it produces a good AUSE."

### Where Form B and the pre-proposal differ

- **Frame selector.** Form B keeps it as a secondary contribution and a floor deliverable, because it is already approved and needs no training ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Updates). The pre-proposal lists it only as a stretch-goal experiment (p. 3, §5.4) and as one possible consumer (p. 1).
- **Datasets.** Form B says the dataset is not decided ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Updates). The pre-proposal names nuScenes for training and evaluation and STF / DENSE as the candidate for heavier weather (p. 3, Table 1).
- **Split and threshold.** Form B names the BEVCar weather split and the AbsRel > 0.10 threshold. The pre-proposal says only "the nuScenes validation split" and "a fixed threshold" (p. 2).
- **Framing.** Form B says the defensible claim on nuScenes is the general one, "we can predict when Depth Anything is wrong", with weather as one reported condition ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Use Case). The pre-proposal's title and target application are adverse driving conditions (p. 1).

### Origin

The head grew out of the team's first plan. Its core contribution was the cross-modality frame selector, a hand-designed rule that trains nothing. Form B makes the learned head the main contribution and keeps the selector as a secondary contribution and a floor deliverable ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Updates; § Secondary contribution: the frame selector).

## Related

- [[Perception in Adverse Conditions]]: how this relates to the CS513 project
- [[CS566-project]]: the area for the project.
- [[Depth Anything V2]]: the frozen backbone.
- [[nuScenes]]: the training and evaluation data.
- [[Sparsification Curve]]: the evaluation method.
- [[HyDRa]], [[LiCROcc]]: related work that uses radar to improve depth or occupancy.
- [[Radar]]: the sensor behind two input channels.
- [[Pretrained Visual Encoders]]: the general pattern of a trained head on a frozen model.
- [[Lidar and Radar Projection into Camera Images]]: the label and radar masks, built and measured.
- [[Confidence Head Input Cache]]: the cached inputs and labels, built on v1.0-mini.
- [[CS566 Implementation Deviations and Open Questions]]: where the code differs from this plan.

## Sources

- [[CS566 Project Pre-proposal]]
- [[CS566 Brainstorm - Project Scope Visual (Form B)]]
- [[Load nuScenes mini and project lidar and radar into the cameras · Spec]]
- [[Load nuScenes mini and project lidar and radar into the cameras · Verification 1]]
- [[Load nuScenes mini and project lidar and radar into the cameras · Verification 4]]
- [[CS566 Project Pre-preposal Feedback]]

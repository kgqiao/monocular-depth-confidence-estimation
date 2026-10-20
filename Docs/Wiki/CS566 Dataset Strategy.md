---
id: doc-kd5aff
type: topic
kind: concept
description: "The CS566 Datasets and Training plan for data: full nuScenes first, to test the confidence head on clean data, then STF / DENSE and ZOD for adverse weather."
tags: [machine-learning/cs566-deeplearning/cs566-project]
aliases: [Dataset Strategy, CS566 dataset plan]
created: 2026-10-12T15:34:33
updated: 2026-10-12T15:39:28
refreshed: 2026-10-19T17:41:36
status: stable
sources: ["[[CS566 Dataset Strategy (plan, 2026-10-12)]]", "[[CS566 Project Pre-preposal Feedback]]", "[[nuScenes]]", "[[Lidar and Radar Projection into Camera Images]]", "[[Autonomous driving datasets]]"]
---

> [!concept] Concept
> 5 sources · refreshed 2026-10-19 · #machine-learning/cs566-deeplearning/cs566-project

## Definition

The dataset plan of the CS566 Datasets and Training role (Nathan Aday, Nathan Bao; see [[CS566 Project Team Roles]]). It names the data that the [[Per-Pixel Depth Confidence Head]] trains and is scored on, in which order, and why. The team wrote it on 2026-10-12, after the TA feedback ([[CS566 Project Pre-preposal Feedback]]). The theory of the head is not repeated here.

## Explanation

### Decision

From [[CS566 Dataset Strategy (plan, 2026-10-12)]], Decision:

1. **Stage 1: full nuScenes v1.0 (trainval and test), not v1.0-mini.** Stage 1 shows whether a confidence head on a frozen [[Depth Anything V2]] can predict where the depth is wrong. nuScenes is well curated: the sensors are calibrated and synchronized, and the lidar is accurate to ±2 cm. So a label error comes from the depth model, not from the data. If the head fails on clean data, adverse-weather data will not make it work.
2. **Stage 2: adverse-weather transfer, after Stage 1 works.** STF / DENSE first, ZOD second. Stage 2 tests whether the confidence drops where the weather is worse (fog, snow, heavy rain). nuScenes has little of that weather.

### Stage 1: nuScenes

The facts below come from the nuScenes paper (Caesar et al., CVPR 2020, Tables 1 and 2) and the devkit split file, as checked in [[CS566 Dataset Strategy (plan, 2026-10-12)]], Stage 1: nuScenes, Facts. See [[nuScenes]] for the rest.

| Item | Value |
|---|---|
| Scenes | 1,000, 20 s each; 700 train, 150 val, 150 test |
| Keyframes (annotated at 2 Hz) | 28,130 train, 6,019 val, 6,008 test; 40,157 in all |
| Keyframe camera images (6 per keyframe) | 168,780 train, 36,114 val, 36,048 test |
| All data, with the sweeps between keyframes | 1.4M camera images, 400k lidar sweeps, 1.3M radar sweeps, 1.4M 3D boxes, 5.5 h, Boston and Singapore |
| Cameras (6) | 12 Hz, 1600 x 900 JPEG; front and side FOV 70°, rear FOV 110° |
| Lidar (1) | Spinning, 32 beams, 20 Hz, ≤ 70 m, ±2 cm, up to 1.4M points/s |
| Radar (5) | 77 GHz FMCW, 13 Hz, ≤ 250 m; no reliable elevation |
| Conditions, all keyframes | 19.4% rain, 11.6% night |
| Conditions, val split | 4,449 day-clear, 1,088 rain, 602 night keyframes (reported by md4all, Gasperini et al., ICCV 2023) |
| Label density, measured on v1.0-mini | About 3,600 lidar pixels (0.25%) and 31 radar points per camera image; 61 of 2,424 images get no radar point |
| Download | About 300 GB or more for the trainval blobs (team estimate; confirm at download) |

The label density row is from [[Lidar and Radar Projection into Camera Images]]. The missing radar elevation is from the TA ([[CS566 Project Pre-preposal Feedback]]).

**What the counts mean for claims.** Synthesis in [[CS566 Dataset Strategy (plan, 2026-10-12)]], to verify with the team's own count: the val condition counts sum to 6,139, so about 120 keyframes are both night and rain. At about 40 keyframes per scene, val holds about 15 night scenes and 27 rain scenes. 602 night keyframes give about 13M labeled lidar pixels, so pixels are not the limit. Pixels in one scene are correlated, so each per-condition claim rests on about 15 to 27 independent scenes.

**Test split.** Its sensor data is public; only the boxes are withheld. So lidar labels can be made on test. md4all states that test has no weather or lighting metadata, so test gives one all-conditions number only.

### Splits

Split by scene, never by keyframe: keyframes in one scene are correlated ([[CS566 Dataset Strategy (plan, 2026-10-12)]], Splits).

| Split | Scenes | Use |
|---|---|---|
| train | Official train minus dev, about 630 | Fit the head |
| dev | About 70 official train scenes, picked with a fixed seed, with night and rain scenes in it | Choose the checkpoint and settings; never reported |
| report | Official val, 150 | All reported numbers, per condition; used for no choice |
| test | Official test, 150 | Optional: one all-conditions number at the end |

Condition labels come from `scene.description`: night if the text says "night", rain if it says "rain". Decide first how to count text such as "Night, after rain" (scene-1094 in mini). Then compare the lists with the BEVCar val weather split ([[Fix the train, validation, and weather splits]]).

### Checks before per-condition claims

The TA asks for label counts per condition before per-condition claims ([[CS566 Project Pre-preposal Feedback]]). Write one table to the repository's `results/`, per split and condition: scenes, keyframes, camera images, labeled lidar pixels, positive-label rate, and radar points per image. Proposed rule: a condition with fewer than 10 scenes in the report split is reported as indicative only, with its scene count next to the number ([[CS566 Dataset Strategy (plan, 2026-10-12)]], Checks before any per-condition claim).

### Stage 2: STF / DENSE, the primary transfer target

From [[CS566 Dataset Strategy (plan, 2026-10-12)]], Stage 2, which checked the STF paper (Bijelic et al., CVPR 2020) and its GitHub release:

- **Purpose:** real fog and snow, and a fog chamber where visibility is set (30, 40, and 50 m, by day and at night). Weather severity becomes a controlled variable: confidence should drop as visibility drops.
- **Size:** about 12,000 annotated real samples and 1,500 annotated chamber samples, from about 10,000 km in Germany, Sweden, Denmark, and Finland. Real labeled frames are about 5.5k clear, 1k dense fog, 1k light fog, and 4k snow or rain; the paper's own totals disagree slightly.
- **Sensors:** stereo camera (1920 x 1024, 30 Hz), gated NIR camera, FIR camera, two lidars (HDL64 S3D and VLP32C, 10 Hz), one proprietary 77 GHz radar (15 Hz), weather station.
- **Fit with the head:** lidar labels from 64 beams, denser than nuScenes. Left-camera history frames at offsets −5 to +3 around each annotated sample, enough for temporal variance. One front radar as a target point cloud. Weather metadata per sample.
- **Shift from nuScenes:** different camera, lidar, and radar. The radar channels need recomputation.
- **Access:** download after registration on the project website. The lead time is not known, so request access now.

### Stage 2: ZOD, the secondary transfer target

From [[CS566 Dataset Strategy (plan, 2026-10-12)]], Stage 2, which checked the ZOD paper (Alibeigi et al., ICCV 2023) and zod.zenseact.com:

- **Purpose:** Nordic weather and night at scale, under a permissive license (CC BY-SA 4.0).
- **Size:** Frames, about 100k images from 14 countries, almost 20k at night; Sequences, 1,473 of 20 s, one annotated middle frame each; Drives, 29 of a few minutes.
- **Sensors:** one front fisheye camera (120°, 3848 x 2168, about 10 Hz); three lidars (one VLS128, two VLP-16); one Continental ARS513 radar.
- **Radar, corrected:** the pre-proposal says ZOD has no radar ([[Autonomous driving datasets]]). The ZOD website says radar was added to Sequences and Drives on January 27, 2025. So ZOD Sequences can feed the radar channels.
- **Fit with the head:** lidar labels, temporal variance, and radar in Sequences; weather labels (clear, cloudy, rainy, foggy, snowy) and road wetness and snow cover.
- **Shift from nuScenes:** a fisheye camera. Depth Anything V2 was not trained on fisheye images, so the images may need rectification first.

### Not used

DAWN has no lidar, radar, or sequences, so it gives no labels and no inputs ([[Autonomous driving datasets]]).

### Open items

1. Download nuScenes trainval and test to CARC, and make the counts table.
2. Request STF / DENSE access now.
3. Fix the night and rain rule, including "after rain".
4. Confirm whether the test split's `scene.description` holds weather text.

## Related

- [[CS566 Confidence Head Training Run 1]]: the first training run on this data.
- [[Per-Pixel Depth Confidence Head]]: the method and its theory.
- [[nuScenes]]: the Stage 1 dataset.
- [[Autonomous driving datasets]]: the other datasets the team considered.
- [[Fix the train, validation, and weather splits]]: the thread that fixes the condition lists.
- [[CS566-project]]: the project.

## Sources

- [[CS566 Dataset Strategy (plan, 2026-10-12)]]: the plan and the checked external facts.
- [[CS566 Project Pre-preposal Feedback]]: the TA's risks on radar and condition coverage.
- [[nuScenes]]: the dataset facts already in the wiki.
- [[Lidar and Radar Projection into Camera Images]]: label and radar density on mini.
- [[Autonomous driving datasets]]: DAWN, and the earlier ZOD entry.

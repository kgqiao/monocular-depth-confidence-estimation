# literature review evidence: [Seeing Through Fog Without Seeing Fog (STF / DENSE)](../../Papers/1902.08913.pdf)

Bijelic et al., CVPR 2020 · bib key `bijelic2020seeing` · report §2.4

## ai-generated summary review

### what the paper does

It introduces a driving dataset recorded over 10,000 km in northern Europe with camera, lidar, radar, gated NIR and FIR sensors, covering fog, snow and rain, plus recordings in a fog chamber at controlled visibilities. It also proposes a single-shot object detector that fuses the sensors adaptively, steered by per-sensor measurement entropy, trained only on clear-weather data and tested on the adverse-weather data.

- p. 1, Abstract, lines 10-15: the dataset, "acquired in over 10,000 km of driving", with "100k labels for lidar, camera, radar, and gated NIR sensors"
- p. 2, right column, §1, first contribution bullet, lines 1-5: sensors and conditions; "heavy fog, heavy snow, and severe rain"
- p. 5, right column, §3.2, paragraph "Controlled Condition Recordings", lines 4-8: fog chamber, "35k frames at a frame rate of 10 Hz", a labeled subset of 1,5k frames, day/night, visibilities "of 30 m, 40 m and 50 m"
- p. 1, Abstract, lines 18-22: the method, "adaptively fuses features, driven by measurement entropy", validated after being "trained on clean data"
- p. 3, left column, §3, ¶1, lines 1-5: the labels are "2D and 3D detection bounding boxes" with a classification of weather, illumination and scene type (the task is object detection, not depth estimation)

### the one result or idea that matters to us

The dataset puts camera, lidar and radar in the same recordings under heavy fog, snow and rain, which the paper's own comparison table says nuScenes does not have, but adverse frames are rare and lidar itself degrades in them. Among the labeled real-world frames, about 1k are dense fog, 1k light fog and 4k snow/rain, and a lidar-only detector falls from 73.46 to 28.98 AP (easy) between clear and dense fog while the image-only detector stays almost steady.

- p. 3, right column, Table 1, columns "NuScenes [6]" and "Ours": rows Heavy Weather and Fog Chamber are a cross for NuScenes and a check for Ours; Ours has 2 lidar sensors, 1 radar sensor, 13.5K labeled frames, 100K labels
- p. 3, right column, §3, ¶1, lines 8-10 of the paragraph (lines 3-5 below Table 1): "also in heavy snow, rain, and fog"
- p. 5, left column, §3.2, paragraph "Real-world Recordings", lines 5-10: "1.4 million frames" collected, "Every 100th frame was manually labeled", giving 5,5k clear, 1k dense fog, 1k light fog, "4k captures in snow/rain"
- p. 4, left column, text below Figure 3 (end of §3, ¶3), lines 1-6: lidar is "highly degraded by scattering media", maximal perceivable distance limited to 25 m at fog densities below 50 m; radar waves "do not strongly scatter in fog [24]" but have low azimuthal resolution
- p. 8, Table 2, rows "Lidar-only SSD" and "Image-only SSD", columns clear and dense fog (easy/mod./hard): lidar 73.46/57.32/54.62 to 28.98/25.24/24.56; image 85.43/75.75/67.79 to 87.89/78.25/74.96. Text at p. 8, left column, first (carried-over) paragraph, lines 6-8 on the page: "it decreases substantially for lidar data"

### how it relates to our design

Our report lists STF / DENSE as the stage-2 candidate for weather nuScenes lacks, and the plan is to request access; the paper confirms it has the lidar, radar and weather annotations our method needs. Two things in the paper bear on using it our way: the radar is a single proprietary sensor that the authors treat as having no vertical information, and the lidar that would supply our labels loses range in fog.

- [literature-review.tex:67](../latex/sections/literature-review.tex:67): the stub for this paper; "fog, snow, and rain with a fog chamber where visibility is controlled; the stage-2 candidate for weather nuScenes lacks; access is by request"
- [literature-review.tex:61](../latex/sections/literature-review.tex:61): what our method needs from a dataset; "lidar, camera sequences, calibrated radar, and condition labels in the same recording". Paper side: p. 5, left column, paragraphs "Radar" (lines 1-4) and "Lidar" (lines 1-2, 7-8): one "(FMCW) radar at 77 GHz", two Velodyne scanners (HDL64 S3D and VLP32C) with ranges of 100 m and 120 m; p. 3, right column, §3, ¶2, lines 1-6: weather annotated manually "at a frame rate of 0.1 Hz", light versus dense fog split at visibility 1 km and 100 m
- [results.tex:13](../latex/sections/results.tex:13): our radar residual is noisy, likely because "nuScenes radar has no elevation". Paper side: p. 5, right column, §4.1, paragraph "Data Representation", lines 13-18: the authors "consider radar invariant along the vertical image axis" and replicate the scan vertically; p. 8, left column, first paragraph, lines 10-11 on the page: the radar input is limited to "a maximum of 100 measurement targets"
- [progress.tex:34](../latex/sections/progress.tex:34): our labels come from lidar returns within 1-80 m. Paper side: p. 2, left column, §1, ¶2 (starting "Existing fusion methods"), lines 21-23: in foggy or snowy conditions pulsed lidar is "restricted to less than 20 m range" due to backscatter. The paper does not discuss lidar as depth ground truth; the consequence for our labels is our reading, not the paper's claim.
- [plan.tex:22](../latex/sections/plan.tex:22): plan item "Request STF / DENSE access". Paper side: p. 1, Abstract, lines 22-24: "Code and data are available here", followed by a GitHub URL (no request procedure is described)

### notes for the writer

- "Access is by request" is not in the paper. The only statement on availability is the Abstract's "Code and data are available here" with the GitHub link; the names "STF" and "DENSE dataset" are also not used for the dataset (DENSE appears only as the EU project name in the Acknowledgment, p. 8). Absence claim, not quotable; cite the project page for access terms if the sentence stays.
- "Weather nuScenes lacks" should be worded as heavy weather and fog, not rain in general: our own report counts nuScenes rain scenes ([results.tex:44](../latex/sections/results.tex:44)), and the paper's Table 1 (p. 3) marks NuScenes with a check for Light Weather and a cross only for Heavy Weather and Fog Chamber. Do not cite that table for nuScenes hardware: it lists 4 radar sensors for NuScenes, where [literature-review.tex:63](../latex/sections/literature-review.tex:63) says 5.
- The paper's own figures disagree in places, so avoid quoting a campaign duration or a single frame total: "three months of acquisition" (p. 2, left column, last paragraph) versus "two months" (p. 4, Figure 2 caption) versus two test drives "for two weeks each" (p. 5, §3.2); Table 1 says 13.5K labeled frames while §3.2 lists 5,5k + 1k + 1k + 4k. Fog-chamber visibilities are 30/40/50 m in §3.2 but Figure 3's caption (p. 4) shows 23 m. Annotation details and the train/test split are in a supplemental document that is not part of this PDF (pp. 9-11 are references only), and the PDF itself carries only the arXiv stamp (v3, 30 Jun 2020), not the CVPR venue.

## post-review discussion with writer

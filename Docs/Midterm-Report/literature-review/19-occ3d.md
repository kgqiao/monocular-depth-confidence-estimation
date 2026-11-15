# literature review evidence: [Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving](../../Papers/2304.14365.pdf)

Tian et al., NeurIPS 2023 (Datasets and Benchmarks Track) · bib key `tian2023occ3d` · report §2.3

## ai-generated summary review

### what the paper does

It defines camera-based 3D occupancy prediction (per voxel: occupied, free or unobserved, plus a semantic class) and builds two benchmarks for it, Occ3D-nuScenes and Occ3D-Waymo, with a label pipeline that works from lidar scans and their annotations in three stages: voxel densification, occlusion reasoning, and image-guided voxel refinement. It also benchmarks existing models and proposes its own, CTF-Occ.

- p. 1, Abstract, lines 6-11: "we develop a label generation pipeline", its three stages, and the two benchmarks "derived from the Waymo Open Dataset and the nuScenes Dataset"
- p. 3, §3.1, ¶1, lines 2-6: the input is "N surround-view camera images" with known intrinsics and extrinsics; the ground truth is the occupancy state (occupied, free, unobserved) and a semantic label per voxel
- p. 3, §3.3, ¶1, lines 1-3: "we take advantage of LiDAR scans and their annotations" because annotating occupancy from images is not possible
- p. 4, §3.3, ¶2 (first full paragraph under Figure 2), lines 2-7: multi-frame aggregation of static and dynamic points separately, K-nearest-neighbor label assignment, mesh reconstruction for hole filling, ray casting "from both LiDAR and camera perspectives", then image-guided refinement
- p. 2, §1, ¶4, lines 1-2: the nuScenes side builds on "nuScenes [4] and Panoptic nuScenes [11]"

### the one result or idea that matters to us

On Occ3D-nuScenes the grid covers 40 m to each side of the ego vehicle and -1 m to 5.4 m in height at 0.4 m voxels (200 × 200 × 16), with 16 classes plus a general-object class, and IoU and mIoU are computed only on voxels marked observed in both the lidar visibility mask and the camera visibility mask. An mIoU number there is therefore a mean of per-class IoUs over the voxels that the aggregated lidar reached and that the current cameras can see; occluded and unscanned voxels do not count.

- p. 3, §3.2, ¶1, lines 3-4: "It has 16 common classes with an additional general object (GO) class" and the range "[-40m, -40m, -1m, 40m, 40m, 5.4m]" with voxel size [0.4m, 0.4m, 0.4m]
- p. 3, Table 1, row Occ3D-nuScenes: modality C&L, 16+GO classes, 1000 sequences, 40,000 frames, volume [200, 200, 16], resolution [0.4, 0.4, 0.4]
- p. 5, §3.3.2, paragraph "Camera visibility mask", lines 2-5: the first occupied voxel along each camera ray is observed, the rest unobserved, and "evaluation is only performed on the" observed voxels in both the lidar and camera views
- p. 5, §3.3.2, paragraph "Aggregated LiDAR visibility mask", lines 6-7: "A voxel is considered visible if it either reflects LiDAR points", or is traversed by a ray; otherwise it is unobserved
- p. 8, §6.1, paragraph "Dataset and Metrics", lines 3-7: the Occ3D-nuScenes scope (-40 m to 40 m in X and Y, -1 m to 5.4 m in Z), voxel size 0.4 m, and "We adopt the metrics of Intersection-over-Union (IoU) and mean" IoU
- p. 9, Table 3 (Occ3D-nuScenes): 17 per-class IoU columns ("others" plus 16 named classes) and one mIoU column; best row CTF-Occ 28.53, BEVFormer 26.88

### how it relates to our design

Occ3D is the benchmark named in our stretch goal: weight the camera-to-voxel lift of an occupancy network by our confidence map and measure mIoU on Occ3D. Its task takes camera images as input and its labels come from lidar, as ours do, but its labels are aggregated over many frames, voxelized, and carry semantic classes, while ours are per-image AbsRel values at single projected lidar returns.

- [plan.tex:47](../latex/sections/plan.tex:47): the stretch goal, "weighting the camera-to-voxel lift of an occupancy network by the confidence map and measuring mIoU on Occ3D", conditional on items 1-5 being done by the week of November 30
- [literature-review.tex:57](../latex/sections/literature-review.tex:57): the stub for this paper, "the occupancy benchmark for the stretch goal; what mIoU measures there"
- [progress.tex:34](../latex/sections/progress.tex:34): our labels; AbsRel at each lidar return within 1-80 m placed on the grid, target AbsRel > 0.10
- p. 3, §3.1, ¶1, lines 2-3: the benchmark input is surround-view camera images (no lidar or radar at input in the task definition)
- p. 4, §3.3.1, paragraph "Multi-frame aggregation", lines 1-7: points are aggregated across frames, dynamic objects in box coordinates and the static scene in global coordinates, giving "a single-frame dense point cloud"
- p. 5, §3.3.2, paragraph "Camera visibility mask", lines 4-5: evaluation restricted to observed voxels, which is the region a camera-lift change such as ours can be scored on
- [literature-review.tex:51](../latex/sections/literature-review.tex:51) and [literature-review.tex:53](../latex/sections/literature-review.tex:53): the HyDRa and LiCROcc paragraphs already quote occupancy mIoU numbers; this paper supplies the definition for the Occ3D one (44.4 for HyDRa, against 28.53 for the best model in this paper's own Table 3)

### notes for the writer

- The paper names the metric but gives no formula. It does not state whether the free class enters the mean, which split Table 3 is evaluated on, or that the "others" column of Table 3 is the general-object class (Table 4 for Waymo labels that column "GO"). Absence claim, not quotable. Safe wording: "mIoU, the mean over semantic classes of the per-class IoU, computed on voxels visible to both lidar and camera"; do not write "mean over 17 classes" as a statement of the paper without saying it is read off Table 3's columns.
- The paper's split counts for Occ3D-nuScenes disagree with each other: p. 3, §3.2 says 600 train / 150 val / 150 test; p. 8, §6.1 says "700 training scenes and 150 validation scenes"; p. 14, Appendix A says 700 / 150 / 150; p. 20, datasheet item 6 says 600 / 150 / 250 and item 12 says 600 / 150 / 200. All agree on 40,000 frames, and p. 20, item 7 says the annotated 2 Hz frames of nuScenes are used. If the report gives a split, cite §6.1 (the one the experiments use) or avoid the count.
- The labels are not from lidar alone: the pipeline also uses the lidar annotations (boxes and semantic labels) and, in the last stage, "semantic segmentation masks of images" (p. 6, §3.3.3 continued, line 1). The paper does not say where those 2D masks come from for nuScenes (Absence claim, not quotable), and its pipeline validation (p. 7, §4.2, lines 1-2) is run on a subset of Occ3D-Waymo only. The bib entry (NeurIPS 2023, Datasets and Benchmarks Track) matches the p. 1 footer.

## post-review discussion with writer

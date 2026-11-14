# literature review evidence: [LiCROcc](../../Papers/2407.16197v1.pdf)

Ma et al., arXiv preprint 2024 (arXiv:2407.16197v1) · bib key `ma2024licrocc` · report §2.3

## ai-generated summary review

### what the paper does

It trains a lidar-camera network (the teacher) for 3D semantic occupancy prediction on nuScenes-Occupancy, then distills it into a radar-only student (R-LiCROcc) and a radar-camera student (RC-LiCROcc) of the same architecture, all fused in bird's-eye view. Three distillation losses do the transfer: CMRD (feature level, cosine similarity through a residual branch), BRD (scene-level affinity matrices) and PDD (KL divergence between teacher and student output distributions).

- p. 1, Abstract, lines 14-19: "we designed three cross-modal distillation" modules, improving both the radar-only and radar-camera settings by distilling the fused lidar-camera features
- p. 2, §I, left column, paragraph starting "Moreover, we have observed", lines 4-8: a "fusion-based KD method" takes cues from a lidar-camera network "(Teacher)" and transfers them to the radar baseline R-SSC-RS and the radar-camera network
- p. 2, §I, left column, paragraph starting "We utilize the same architecture", lines 1-7: same architecture for both fusion networks; the three modules named in full (Cross-Model Residual Distillation, BEV Relation Distillation, Predictive Distribution Distillation)
- p. 4, §III-C.1 (left column, ¶1, lines 6-8), §III-C.2 (right column, ¶1, lines 6-8) and §III-C.3 (right column, ¶1, lines 1-3): the three losses; "directly forcing radar features to mimic" is expected not to work well (CMRD), "similarity-based affinity matrix" (BRD), "utilizing KL divergence" (PDD)
- p. 5, §IV-A, left column, paragraph "Datasets", lines 1-3: nuScenes-Occupancy, "850 scenes (700 for training and 150 for validation)", 34,000 keyframes

### the one result or idea that matters to us

Lidar is used only while training: it is an input to the teacher, and at inference only the radar or radar-camera student runs. The distillation gain is real but small next to the gap to the teacher: radar-only mIoU goes from 7.5 to 8.5 and radar-camera from 15.6 to 17.1, against 24.7 for the lidar-camera teacher, and the gain is largest in sunny daytime and shrinks in rain and at night.

- p. 3, Fig. 2 caption, line 4: "Inference is performed using only the RC fusion Network", with the camera input optional
- p. 4, §III-C, left column, first two text lines below the Fig. 3 caption (continuation of the paragraph that starts on p. 3): the "image branching of the student model is optional"
- p. 6, Table III (right column, top): R-LiCROcc mIoU 7.5 with no distillation, 7.9 with CMRD, 8.5 with CMRD + PDD; RC-LiCROcc 15.6, 16.1, 16.3, 17.1 as CMRD, BRD, PDD are added
- p. 5, Table I, rows "LC-Fusion (Ours)", "R-LiCROcc (Ours)", "RC-LiCROcc (Ours)", columns IoU / mIoU: 34.9 / 24.7, 21.3 / 8.5, 26.0 / 17.1
- p. 7, right column, paragraph starting "Under sunny daytime conditions", lines 1-2, continuing on p. 8, left column, lines 1-6: "the distillation effect yields the highest performance" in sunny daytime; in rain and at night the gains are less pronounced, and R-LiCROcc "is slightly diminished in rainy weather" (Table V, p. 7: mIoU 7.10, a change of -0.05)

### how it relates to our design

Both systems keep lidar out of the deployed model and keep radar in it, on the same dataset (nuScenes). The mechanism differs: LiCROcc passes lidar knowledge through a trained teacher network's features and predictions to improve the student's occupancy output, while our lidar is projected into the image to produce error labels for a head that scores a frozen depth model and does not change it.

- p. 3, Fig. 2 caption, line 4: "Inference is performed using only the RC fusion Network" (lidar absent at inference)
- p. 3, §III-B, left column, paragraph "Point branch", lines 11-14: radar input is a point cloud of xyz, "radar cross section" and ego-motion-compensated xy velocities; p. 5, §IV-B, right column, ¶1, lines 7-9: "radar scans stitched from 5 radar sensors" (our radar input is a residual and a validity mask on an image grid, not a point-cloud branch)
- p. 5, §IV-B, right column, ¶1, lines 5-7: "we project the point cloud onto the camera view to provide depth supervision" for the LSS image branch during training (the one place the paper uses points projected into the image as a training signal, as our labels do; see the note below on which point cloud)
- [introduction.tex:9](../latex/sections/introduction.tex:9): "Lidar is present when the model is built and absent when it runs; radar, which a production car does carry, is present in both."
- [literature-review.tex:53](../latex/sections/literature-review.tex:53): the stub for this paper; "the teacher-student view of lidar as training-time-only supervision is the same stance our labels take"
- [literature-review.tex:47](../latex/sections/literature-review.tex:47) and [literature-review.tex:75](../latex/sections/literature-review.tex:75): we use radar to tell where the frozen depth is wrong, "so the output is a confidence map rather than a better depth map"; the fusion literature uses radar to improve the prediction itself, which is what LiCROcc does for occupancy

### notes for the writer

- "Same stance" holds only at the level of "lidar at training, not at inference". In LiCROcc lidar is a teacher network's input and the student imitates the teacher's features and output distribution; the paper never uses lidar as a direct per-pixel or per-cell label for the student. Also, our own pipeline is not yet as lidar-free at inference as LiCROcc's: [plan.tex:51](../latex/sections/plan.tex:51) says the lidar fit "uses a sensor the deployed system lacks".
- The abstract's headline gains (22.9%, 44.1%, 15.5% mIoU "exceeding the baseline", p. 1, Abstract, lines 19-22) are relative improvements over other methods, not the effect of distillation. The text names M-CONet as the baseline for the first figure (p. 6, §IV-C, left column, lines 3-6 below Table II, in the paragraph continued from p. 5: "a 23% increase in mIoU"); the other two match R-CONet (5.9) and RC-CONet (14.8) in Table I only by my arithmetic, the paper does not say so. The distillation effect is the Table III figure: +1.0 mIoU for radar-only (13.3% relative, p. 6, §IV-C, left column, ¶2, lines 6-8) and +1.5 mIoU for radar-camera (p. 6, same paragraph, lines 17-19). BRD is applied only to the radar-camera student (p. 4, §III-C.2, right column, last paragraph, lines 7-8: "BRD has only been used to distill"), so "CMRD, BRD, PDD" all three apply to RC-LiCROcc only.
- Two wording traps. The paper spells the third module both "PDD" (abstract, §III-C.3, Table III) and "PPD" (p. 2, §I, twice); use PDD. And the depth-supervision sentence on p. 5 says "the point cloud" without saying lidar or radar, so do not write that LiCROcc supervises depth with lidar as a quoted fact. `references.bib` lists the paper as an arXiv preprint, which matches the PDF (arXiv stamp "23 Jul 2024", no venue printed).

## post-review discussion with writer

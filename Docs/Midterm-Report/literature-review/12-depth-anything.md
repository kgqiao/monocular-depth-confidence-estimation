# literature review evidence: [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](../../Papers/2401.10891.pdf)

Yang et al., CVPR 2024 · bib key `yang2024depthanything` · report §2.1

## ai-generated summary review

### what the paper does

It trains a teacher depth model on 1.5M labeled images from six public datasets, uses it to pseudo-label about 62M unlabeled images from eight public datasets, and trains a student jointly on the labeled and pseudo-labeled images. The network is a DINOv2 encoder with a DPT decoder, trained like MiDaS with an affine-invariant loss in disparity space, and is released in three sizes (ViT-S, ViT-B, ViT-L).

- p. 1, Abstract (left column), lines 5-8: the data engine that annotates "large-scale unlabeled data (∼62M)" to enlarge data coverage
- p. 2, §1, ¶3 (left column, starts "We design a data engine"), lines 3-10: 62M images "from eight public large-scale datasets"; 1.5M labeled images from six datasets train an initial model; unlabeled images are then annotated and learned jointly "in a self-training manner"
- p. 3, §3, ¶1 (left column), lines 4-8: "learn a teacher model T", assign pseudo depth labels, then train a student S on the combination of both sets
- p. 5, §4.1, ¶1, lines 1-3 (left column bottom into right column top) and lines 4-10 (right column): "we use the DPT [47] decoder" on a DINOv2 encoder; teacher trained 20 epochs on labeled images; pseudo-labels come from "a best-performed teacher model" with a ViT-L encoder; labeled to unlabeled ratio 1:2 per batch
- p. 5, Table 2 caption: three model scales, ViT-S (24.8M), ViT-B (97.5M), ViT-L (335.3M)

### the one result or idea that matters to us

Adding the 62M pseudo-labeled images did not help by itself; the gain appeared only when the student had to learn the clean teacher's labels from strongly perturbed inputs (color jitter, Gaussian blur, CutMix), plus a feature-alignment loss to a frozen DINOv2 encoder. The paper presents that as the source of its robustness: zero-shot on six unseen datasets, including the driving sets KITTI and DDAD, it beats the best MiDaS v3.1 model, and its ViT-B already beats that larger ViT-L MiDaS.

- p. 4, §3.2, paragraph starting "Unfortunately, in our pilot studies" (left column), lines 1-2: "we failed to gain improvements with such a self-training pipeline"
- p. 4, §3.2, paragraph starting "To address the dilemma" (left column), lines 3-11: "We inject strong perturbations to unlabeled images"; the two forms are color distortions (color jittering, Gaussian blurring) and CutMix; this helps the model "deal with the open world more robustly"
- p. 4, §3.2, paragraph starting "We use CutMix with 50% probability" (right column), lines 2-4: images given to the teacher for pseudo-labeling "are clean, without any distortions"
- p. 8, Table 9 (left column), AbsRel on KITTI / DDAD: labeled only 0.085 / 0.245; plus unlabeled without perturbation 0.085 / 0.242 (and worse on NYUv2, ETH3D, DIODE); plus perturbations 0.081 / 0.235; plus feature alignment 0.076 / 0.230. Discussed on p. 7, §4.5, paragraph "Effectiveness of 1) challenging the student model", right column, lines 2-5: "learn more robust representations"
- p. 5, Table 2, rows MiDaS v3.1 ViT-L and Depth Anything ViT-B: KITTI AbsRel 0.127 versus 0.080, δ1 0.850 versus 0.939; DDAD AbsRel 0.251 versus 0.232, δ1 0.766 versus 0.786. p. 5, §4.2, ¶2 (right column), lines 1-2: "our ViT-B model is already clearly superior"

### how it relates to our design

This is the first version of our backbone, and the properties our design depends on originate here: the DINOv2 encoder with a DPT decoder that V2 keeps, and a relative (affine-invariant disparity) output that has to be aligned in scale and shift to ground truth before an error can be measured, which the authors themselves do at evaluation. It makes depth more robust by widening the training data and offers no per-pixel reliability output, which is the gap our report describes.

- p. 5, §4.1, ¶1, lines 1-3: "We adopt the DINOv2 encoder [43] for feature extraction" and the DPT decoder; [literature-review.tex:27](../latex/sections/literature-review.tex:27) says V2 "keeps the DINOv2 encoder and DPT decoder of its predecessor"
- p. 3, §3.1, ¶1 (left column), lines 3-7: depth is transformed "into the disparity space by d = 1/t", and the affine-invariant loss ignores "unknown scale and shift of each sample"; p. 2, §2, paragraph "Zero-shot depth estimation", ¶2 (right column), lines 12-13: "we still follow MiDaS in relative depth estimation". Compare [literature-review.tex:19](../latex/sections/literature-review.tex:19): the family's output is "affine-invariant disparity rather than metric depth"
- p. 9, §6 (supplementary), ¶1 (left column), lines 7-9: in zero-shot evaluation "the scale and shift of our prediction are manually aligned with the ground truth"; [progress.tex:11](../latex/sections/progress.tex:11): our two per-image least-squares fits of disparity to inverse depth (lidar and radar)
- p. 9, §8 (right column), lines 1-7: the largest model is ViT-Large and the authors plan a ViT-Giant teacher "producing more accurate pseudo labels for smaller models", which is the step V2 takes ([literature-review.tex:27](../latex/sections/literature-review.tex:27)); [progress.tex:11](../latex/sections/progress.tex:11) is the V2 ViT-B checkpoint we actually run, not this paper's
- Absence claim, not quotable: no confidence or uncertainty output is described anywhere in pp. 1-18; "confidence" appears only in a reference title, and "uncertainty" not at all. [introduction.tex:7](../latex/sections/introduction.tex:7) states the gap: every pixel gets a depth value "and no measure of reliability"

### notes for the writer

- Do not carry V2's recipe back to V1. Here the teacher is a ViT-L trained on 1.5M labeled real images and the student learns from labeled and pseudo-labeled images together (p. 5, §4.1). The synthetic-only ViT-G teacher and students trained purely on pseudo-labels are V2. The paper's own term is "self-training" with a "teacher model" and "student model", so "teacher-student setup" is fair.
- "What it changed about robustness" is supported in a specific sense: zero-shot accuracy on six unseen datasets (Table 2) and the Table 9 ablation. Robustness to adverse conditions is only qualitative: the Figure 1 caption (p. 1) says the model "works robustly in low-light environments" and names foggy weather. Absence claim, not quotable: no table reports accuracy split by lighting or weather, so do not cite this paper for measured night or rain performance.
- `references.bib` gives CVPR 2024. The PDF in Docs/Papers is the arXiv version (margin stamp "arXiv:2401.10891v2 [cs.CV] 7 Apr 2024") and does not print a venue, so the venue cannot be checked against this PDF. Title and the six authors match.

## post-review discussion with writer

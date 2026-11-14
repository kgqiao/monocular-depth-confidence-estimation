# literature review evidence: [On the Uncertainty of Self-Supervised Monocular Depth Estimation](../../Papers/2005.06209.pdf)

Poggi et al., CVPR 2020 · bib key `poggi2020uncertainty` · report §2.2

## ai-generated summary review

### what the paper does

It compares 11 ways of estimating per-pixel uncertainty for a self-supervised monocular depth network (Monodepth2 on KITTI, trained with monocular, stereo, or both kinds of supervision): one flip-based (Post), three empirical (Drop, Boot, Snap), three predictive (Repr, Log, Self) and four combinations. It also proposes one of them, Self-Teaching (Self), where a second copy of the network is trained on the first copy's depth output and predicts depth plus a variance.

- p. 1, Abstract, lines 7-10: "we explore for the first time how to" estimate uncertainty for this task, and propose a new technique
- p. 2, left column, §1, the three contribution bullets: "A comprehensive evaluation of uncertainty estimation", the study of the training paradigm, and the Self-Teaching paradigm
- p. 6, left column, §4.1, paragraph "Architecture and training schedule", lines 1-2: "baseline model Monodepth2 [20]"; paragraph "Dataset" (bottom of the same column), lines 1-2: KITTI, "made of 61 scenes"
- p. 7, Tables 1 and 2, and p. 8, Table 3: the 11 uncertainty rows (Post, Drop, Boot, Snap, Repr, Log, Self, Boot+Log, Boot+Self, Snap+Log, Snap+Self) under M, S and MS supervision
- p. 8, right column, §5, lines 5-6: "up to 11 strategies"

### the one result or idea that matters to us

The evaluation protocol. Pixels of each depth map are sorted by descending uncertainty and removed 2% at a time, and the error of the remaining pixels is plotted (a sparsification curve). AUSE is the area between that curve and the oracle curve (pixels sorted by true error), lower is better; AURG is the area between the random (flat) curve and that curve, higher is better. The error is Abs Rel, RMSE or δ ≥ 1.25.

- p. 6, right column, §4.1, paragraph "Uncertainty metrics", lines 1-3: "we use sparsification plots as in" [27] (Ilg et al., ECCV 2018, reference list p. 9)
- same paragraph, lines 3-7: sort "in order of descending uncertainty", remove a subset "(i.e., 2% in our" experiments), compute the error on the remaining pixels
- same paragraph, lines 8-12: the oracle sorts by error magnitude; a random uncertainty is "modelled as a constant", giving a flat curve
- same paragraph, lines 12-18: "we can measure the Area Under the" Sparsification Error (AUSE, lower is better); "the Area Under the Random Gain" (AURG, higher is better), random minus estimated
- same paragraph, lines 21-22: "We assume Abs Rel, RMSE or δ ≥ 1.25" as the error metric

### how it relates to our design

We report AUSE and AURG with AbsRel as the error, which is one of the paper's three variants, and we plan flip disagreement as a baseline, which is the paper's Post: the absolute difference between the depth of an image and the back-flipped depth of its mirrored copy, usable on any pretrained model with two forward passes. Self is not a head on a frozen model: it retrains a full second network, so it conflicts with our frozen-backbone constraint; the paper's nearest description of our head is a sentence in the Repr paragraph about classifiers trained on the prediction error, which it says cannot be used without ground truth.

- p. 4, left column, §3.1, ¶1, lines 1-7 and Eq. 5: Post, after Godard et al. [19]; depth from the image and from "its horizontally flipped" counterpart; uncertainty is the absolute difference of the two (u_Post = |d − back-flipped d|)
- p. 4, left column, §3.1, text after Eq. 5, lines 3-6: needs 2× forwards at test time, but "it can be applied seamlessly to any" pretrained framework without modification
- p. 5, right column, §3.3, paragraph "Self-Teaching (Self)", lines 4-7 and Eq. 14: a first network T is trained self-supervised, then "we train a second instance of the same model", S, on T's depth with a negative log-likelihood loss (predicted mean and variance)
- p. 6, left column, §4.1, paragraph "Architecture and training schedule", lines 15-17: for predictive models (Repr, Log, Self) "a single output channel is added in parallel" to the depth channel
- p. 5, left column, §3.3, paragraph "Learned Reprojection (Repr)", lines 1-6: learning a function over "the prediction error employing a classifier is a popular" technique for stereo and optical flow, not applicable here "given the absence of ground truth labels"
- [literature-review.tex:31](../latex/sections/literature-review.tex:31): the backbone is frozen, so "we are closest to the post-hoc methods that read a trained model's behavior without retraining it"; sparsification curves and AUSE come from this literature
- [results.tex:77](../latex/sections/results.tex:77) and [results.tex:83](../latex/sections/results.tex:83): results table caption "AUSE (lower is better) and AURG (higher is better)"; curves are AbsRel of the remaining cells against the fraction removed, with random removal and the oracle
- [progress.tex:48](../latex/sections/progress.tex:48) and [plan.tex:15](../latex/sections/plan.tex:15): the flip-augmentation baseline is not built yet; plan is to run the backbone on horizontally flipped val keyframes, align with a lidar-free scale-and-shift fit, and score the per-cell relative difference

### notes for the writer

- The todo at [literature-review.tex:41](../latex/sections/literature-review.tex:41) calls Self a "learned-confidence head that is the closest relative of ours". The paper does not support that wording. Self is a whole student network retrained to output depth and variance, supervised by the teacher's depth, with no ground truth and no frozen model (p. 5, right column, Self-Teaching paragraph; p. 7, Table 1a, #Trn column "(1+1)×"). What is safe to say: Self is the paper's learned (predictive) uncertainty and its best method when pose is unknown (p. 7, left column, §4.2, paragraph "Uncertainty", lines 7-9: "outperforms Log by a notable margin"; Table 1b, Abs Rel AUSE 0.030 for Self versus 0.039 Log and 0.044 Post). With stereo supervision Log is better (Table 2b, RMSE AUSE 0.938 Log versus 1.679 Self). The kinship with our head is closer in the Repr sentence cited above, and the paper only mentions that idea, it does not evaluate a ground-truth-trained error classifier. Absence claim, not quotable: no method in pp. 1-8 trains a separate confidence module on a frozen depth network with ground-truth error labels.
- Our planned flip baseline differs from Post in two ways the paper does not cover: we plan a relative difference and a scale-and-shift alignment ([plan.tex:15](../latex/sections/plan.tex:15)), the paper uses the plain absolute difference of two depth maps (Eq. 5) with no alignment step. Say "adapted from", not "identical to". Also, the paper gives AUSE and AURG in words only (no formula, no statement of how the area is integrated or normalised), sorts per depth map, and reports Fig. 5 as "the RMSE sparsification error curves, averaged over" the test set (p. 8, left column, §4.5, lines 2-3); any extra detail of our implementation (grid cells, pooling, step size) is ours, not the paper's.
- All numbers are for Monodepth2 on KITTI (Eigen split, improved ground truth, 80 m cap; p. 6, right column, top). Nothing in the paper is measured on nuScenes or on a supervised or foundation depth model, so its rankings should not be presented as predictions for our setting. Minor: the PDF in Docs/Papers is arXiv v1 and carries no CVPR marking (absence claim, not quotable), and its title is printed in sentence case; the bib's CVPR 2020 venue cannot be checked from this PDF.

## post-review discussion with writer

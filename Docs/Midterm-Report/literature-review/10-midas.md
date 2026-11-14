# literature review evidence: [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer (MiDaS)](../../Papers/1907.01341.pdf)

Ranftl et al., TPAMI 2022 (PDF is arXiv v3, header dated 2020) · bib key `ranftl2020midas` · report §2.1

## ai-generated summary review

### what the paper does

It trains one monocular depth network on five datasets whose depth annotations are incompatible (metric depth, depth without scale, stereo disparity without scale or shift), including a new dataset built from 3D movies, by using a loss that ignores each image's scale and shift and a multi-objective strategy for mixing the datasets. It evaluates by zero-shot cross-dataset transfer, on six datasets never seen in training, and reports beating prior methods there.

- p. 1, Abstract, lines 3-5: "enable mixing multiple datasets during training, even if their annotations are incompatible", then a training objective "invariant to changes in depth range and scale"
- p. 1, Abstract, lines 6-8: "five diverse training datasets, including a new, massive data source: 3D films" and the definition of zero-shot cross-dataset transfer ("not seen during training")
- p. 3, Table 1, columns Annotation and Depth, top block: DIML Indoor is "Metric", MegaDepth "No scale", and ReDWeb, WSVD and 3D Movies "No scale & shift"
- p. 9, left column, paragraph starting "Tables 6 and 7 show", lines 9-10: "Combining all five datasets with Pareto-optimal mixing yields our best-performing model"
- p. 13, left column, Table 11, rows "Ours" and below: the MIX 5 model has the best average rank (2.0) across DIW, ETH3D, Sintel, KITTI, NYU and TUM

### the one result or idea that matters to us

The network predicts in disparity space, that is inverse depth known only up to a per-image scale and shift, because the loss first aligns prediction and ground truth in scale and shift and so never penalises either. The authors themselves fit scale and shift per image, in inverse depth and by least squares, before measuring any error against ground truth.

- p. 5, right column, paragraph "Scale- and shift-invariant losses", lines 1-3: "diction in disparity space (inverse depth up to scale and shift)", together with a family of scale- and shift-invariant dense losses
- p. 5, right column, paragraph starting "We identify three major challenges", lines 3-7: the scale ambiguity and the shift ambiguity; "disparity only up to an unknown scale and global disparity shift"
- p. 5, right column, paragraph starting "The first approach aligns", lines 1-2, and Eq. 2 with the two lines after it: alignment "based on a least-squares criterion", with s and t found "in closed form" (Eq. 3-4)
- p. 7, §6, paragraph "Test datasets and metrics", right column lines 6-9: "ground truth in scale and shift for each image before measuring", then "We perform the alignment in inverse-depth space"
- p. 6, left column, Eq. 7 and the two lines before it, with p. 8, left column, paragraph "Comparison of loss functions", lines 9-11: the loss actually used is the trimmed one ("Um = 0.8M", i.e. the 20% largest residuals per image are dropped) and "yields the lowest validation error over all datasets"

### how it relates to our design

Our frozen backbone has the same kind of output, disparity with unknown scale and shift, and we recover the two numbers the same way the paper does at evaluation time: a per-image least-squares fit in inverse depth, in our case against lidar (for the labels) and against radar. The paper gives no confidence or uncertainty output and processes each frame alone, so the error signal our head predicts and our temporal channel are not in it.

- p. 7, §6, paragraph "Test datasets and metrics", right column lines 6-9: per-image scale and shift alignment in inverse-depth space by least squares, "For all our models and baselines"
- p. 5, right column, Eq. 2: (s, t) = arg min Σ (s·d_i + t − d*_i)², the same form of fit we apply to lidar and radar inverse depth
- [progress.tex:11](../latex/sections/progress.tex:11): backbone paragraph; output is "disparity with unknown scale and shift", two per-image least-squares fits of disparity to inverse depth (lidar for labels and the temporal channel, radar for the radar residual)
- [progress.tex:34](../latex/sections/progress.tex:34) and [progress.tex:59](../latex/sections/progress.tex:59): labels are AbsRel of the fitted prediction at lidar returns; deviations row D1, "Relative ViT-B with per-image lidar and radar fits"
- Absence claim, not quotable: the words "uncertainty" and "confidence" do not occur anywhere in pp. 1-14. The only mention of time is p. 9, right column, paragraph starting "We show qualitative results on the DAVIS", lines 3-4: "every frame was processed individually", with no temporal information used.

### notes for the writer

- The paper never says in one sentence that the output "needs a per-image fit before it is in meters". What it says is that prediction is inverse depth up to scale and shift (p. 5), that the stereo-derived training data "does not provide metric depth" (p. 3, §4, ¶1, right column line 6), and that errors in depth space are measured only after a per-image least-squares alignment (p. 7). Write the claim as that chain, and note the paper's fit is against dense ground truth for evaluation, not a deployment recipe.
- [literature-review.tex:19](../latex/sections/literature-review.tex:19) describes the family as having a dense-prediction-transformer decoder. That is not this paper: its network is the "ResNet-based [56] multi-scale architecture" of Xian et al. (p. 6, §6, ¶1, lines 1-3) with a ResNeXt-101-WSL encoder (p. 8, left column, paragraph "Comparison of encoders", last sentence). Also do not say it is best on driving data: on KITTI its zero-shot δ>1.25 is 23.90 against 5.53 for KITTI-trained Godard (p. 13, Table 11); the claim is zero-shot generalization only.
- Bibliography: the key says 2020 and `references.bib` gives year 2022 in TPAMI. The PDF is the arXiv v3 preprint (stamp "25 Aug 2020") with running header "IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. XX, NO. XX, 2020", so the journal is confirmed but the 2022 volume year cannot be verified from this PDF. The name "MiDaS" appears only in the code URL at the end of §7 (p. 13), not in the title or body.

## post-review discussion with writer

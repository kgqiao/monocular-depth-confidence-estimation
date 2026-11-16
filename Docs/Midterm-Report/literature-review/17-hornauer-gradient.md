# literature review evidence: [Gradient-based Uncertainty for Monocular Depth Estimation](../../Papers/2208.02005.pdf)

Hornauer and Belagiannis, ECCV 2022 · bib key `hornauer2022gradient` · report §2.2

## ai-generated summary review

### what the paper does

It estimates per-pixel uncertainty for an already trained depth network without changing its weights and without any training: the image and its horizontal flip are both passed through the network, the squared difference of the two depth maps is used as an auxiliary loss, and that loss is back-propagated to one decoder feature map. The gradient of that feature map, max-pooled over channels, upsampled and min-max normalized to [0, 1], is the uncertainty map.

- p. 1, Abstract, lines 5-11: post hoc approach for an "already trained and thus fixed depth estimation model"; gradients from an auxiliary loss built on the image and its "horizontally flipped counterpart", so no ground truth is needed
- p. 4, §3, paragraph "Post Hoc Uncertainty Estimation", lines 2-4: "We do not adjust the model parameters", only access to the feature maps of the i-th layer of the depth decoder is assumed
- p. 6, top paragraph (continued from p. 5), lines 3-9 and Eq. 1: "we conduct two forward passes" (original and flipped image), then the derivative of the loss with respect to the feature map
- p. 7, §3.2, paragraph "Depth Estimation Model", Eq. 4: the auxiliary loss is the squared difference between the depth and the flipped-back reference depth
- p. 6, §3.1, paragraph "Uncertainty Score", Eq. 2-3 and the text between them: max-pooling over channels, upsampling "by nearest-neighbor interpolation", and after Eq. 3 "normalized to a" range of [0, 1] using the map's own minimum and maximum

### the one result or idea that matters to us

With the same two forward passes as the plain flip difference (Post) plus one backward pass, the gradient map ranks errors better than Post on a fixed Monodepth2: on KITTI, AUSE in Abs Rel falls from 0.044 to 0.029 (monocular supervision) and from 0.036 to 0.022 (stereo), and AUSE in RMSE from 2.864 to 0.540 and from 2.523 to 0.510, while inference time rises from 11.78 ms to 18.84 ms. The method needs no trained model, no specialized training and no ground truth; when ground truth is used for the loss instead of the flip (an ablation only, not usable at inference) the Abs Rel AUSE on NYU drops from 0.061 to 0.020.

- p. 10, Table 1, block "Post", rows "Base [9]" and "Ours": Abs Rel AUSE 0.044 vs 0.029, RMSE AUSE 2.864 vs 0.540, Inf [ms] 11.78 vs 18.84 (KITTI, monocular supervision)
- p. 11, Table 2, block "Post", rows "Base [9]" and "Ours": Abs Rel AUSE 0.036 vs 0.022, RMSE AUSE 2.523 vs 0.510 (KITTI, stereo supervision)
- p. 12, Table 4, row "Ours": #Train "-", Specialized Training "No", #Forward 2, #Backward 1; text below the table, lines 10-11: "our approach only needs one additional backward pass"
- p. 2, §1, paragraph starting "One approach to training-free", lines 5-7: gradients are less costly than sampling and the method "requires only one additional backward pass"
- p. 14, Table 5, rows "GT" and "Flip" (Abs Rel AUSE 0.020 vs 0.061), and §4.3 paragraph on that page, lines 9-10: "the ground truth is not available during inference"

### how it relates to our design

Both methods leave the depth model's weights untouched, read the depth decoder, and are scored with AUSE and AURG. The differences the paper lets us state: it is training-free and never uses ground truth, it adds a flipped forward pass and a backward pass through the depth network for every image, it reads the gradient at one decoder layer (the 6th decoder layer of Monodepth2) rather than the features themselves, and it is tested only on Monodepth2 on KITTI and NYU Depth V2. Our head is trained on lidar error labels and reads decoder features from a frozen Depth Anything V2 that is run once per image.

- p. 9, §4.1, paragraph "Implementation Details", line 15: "we choose the 6th layer of the decoder" to extract the gradients, for all setups; lines 16-17 send the layer comparison to the supplementary material
- p. 8, §4.1, paragraph "Models", lines 1-7: "We choose Monodepth2 [9] as the depth estimation model", trained with monocular and stereo supervision on KITTI and supervised on NYU Depth V2; paragraph "Datasets" on the same page, lines 2-9: KITTI Eigen split (80 m cap) and NYU Depth V2 (10 m cap)
- p. 8, §4.1, paragraph "Evaluation Metrics", lines 8-17: "Area Under the Sparsification Error (AUSE)" and "Area Under the Random Gain (AURG)", each in terms of Abs Rel, RMSE and δ ≥ 1.25, following Poggi et al.
- p. 2, §1, first paragraph on the page, lines 4-6: the motivation, parameters of "models provided by external sources are usually not accessible" and so not modifiable
- [literature-review.tex:31](../latex/sections/literature-review.tex:31): "the backbone is frozen, so we are closest to the post-hoc methods that read a trained model's behavior without retraining it"; [literature-review.tex:75](../latex/sections/literature-review.tex:75): "we train only a small head on top of a frozen one, which is the situation of everyone who uses a public depth checkpoint"
- [introduction.tex:9](../latex/sections/introduction.tex:9): the head outputs a per-cell probability and learns from lidar labels; [progress.tex:11](../latex/sections/progress.tex:11): backbone "frozen, run once per image"; [progress.tex:23](../latex/sections/progress.tex:23): decoder features come from the `output_conv1` tap, 64 channels
- [progress.tex:48](../latex/sections/progress.tex:48), [results.tex:71](../latex/sections/results.tex:71) and [plan.tex:15](../latex/sections/plan.tex:15): our planned flip-disagreement baseline, which corresponds to the paper's Post base method (the row its gradient method is compared against), not to the gradient method itself

### notes for the writer

- "Nearest prior work" in the todo at [literature-review.tex:43](../latex/sections/literature-review.tex:43) is our judgment, not something the paper says. What the paper supports is "post hoc, weights not adjusted, uses a decoder layer". It also names two other post hoc, training-free methods (inference-only dropout from Mi et al. [18] and test-time augmentation variance, p. 12, text below Table 4, lines 6-7: "only In-Drop, Var and our method are post hoc"), so "one of the nearest" or "the nearest we found" is safer. The paper's own word is "fixed" or "already trained", not "frozen". Absence claim, not quotable: nothing in pp. 1-15 trains a separate module on a fixed depth network with ground-truth error labels, so the paper offers no direct comparison between a gradient method and a trained head.
- Do not overstate the scope. Every number is for Monodepth2; nuScenes, DPT and transformer backbones do not appear (absence claim, not quotable). The "6th layer" is specific to Monodepth2's decoder and its justification is in a supplementary file that is not part of this PDF. The gradient method does not win everywhere: on NYU it is worse than the test-time augmentation variance in RMSE for the Post model (p. 12, Table 3, RMSE AUSE 0.252 Ours vs 0.221 Var) and worse than the Log model's own variance in RMSE (0.176 vs 0.159); the paper's own summary says it improves "for most of the cases" (p. 11, paragraph "Summary", lines 1-2).
- The uncertainty map is rescaled per image by its own minimum and maximum (p. 6, Eq. 3), so it is a ranking score within an image; the paper reports only AUSE, AURG and inference time, and no calibration measure (absence claim, not quotable). The PDF in Docs/Papers is arXiv v1 (3 Aug 2022) and carries no ECCV marking, so the bib's ECCV 2022 venue cannot be checked from it; the PDF prints the title as "Gradient-based" (lowercase b) and the authors as Julia Hornauer and Vasileios Belagiannis.

## post-review discussion with writer

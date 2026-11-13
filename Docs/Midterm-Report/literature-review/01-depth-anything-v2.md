# literature review evidence: [Depth Anything V2](../../Papers/2406.09414.pdf)

Yang et al., NeurIPS 2024 · bib key `yang2024depthanythingv2` · report §2.1

## ai-generated summary review

### what the paper does

It trains a large ViT-G teacher only on synthetic images with exact depth labels, uses that teacher to pseudo-label 62M unlabeled real images, and trains the released student models (ViT-S, B, L, G) only on those pseudo-labels. The result is finer detail and more robust depth than V1, at more than 10× the speed of diffusion-based models.

- p. 1, Abstract, lines 4-7: the three practices, starting at "replacing all labeled real images"
- p. 6, §5.1, the three bullets and the line after them: teacher on synthetic, pseudo-label real, train students; "four student models" (small, base, large, giant)
- p. 6, §5.2, ¶1, lines 1-2: "five precise synthetic datasets (595K images)" and eight pseudo-labeled real datasets (62M images)
- p. 9, §7.4, ¶2, lines 5-6: students are trained "purely on pseudo-labeled images"
- p. 1, Abstract, lines 7-10: "more than 10× faster" than Stable Diffusion models, sizes from 25M to 1.3B parameters

### the one result or idea that matters to us

The released models output affine-invariant inverse depth: disparity with unknown scale and shift per image, not meters. Metric depth exists only as separate checkpoints fine-tuned on synthetic data (Hypersim indoor, Virtual KITTI outdoor).

- p. 6, §5.2, ¶1, line 4: "our models produce affine-invariant inverse depth"
- p. 6, footnote 2: metric models come from further fine-tuning, pointing to §7.3
- p. 8, §7.2, ¶1, line 1: "Since our model predicts affine-invariant inverse depth"
- p. 9, §7.3, ¶2, lines 3-5: fine-tuned on "Hypersim [58] and Virtual KITTI [9]" for indoor and outdoor metric depth
- p. 8, Table 2, row Depth Anything V2 / ViT-B, KITTI columns: zero-shot AbsRel 0.078, δ1 0.939 (the checkpoint size we use, on the closest driving benchmark)

### how it relates to our design

We freeze the relative ViT-B checkpoint and read its DPT decoder features, so before an error label can be computed we fit each image's disparity to inverse lidar depth by least squares. That is deviation D1: the proposal named the Virtual KITTI metric checkpoint, which needs no fit. The paper offers no per-pixel confidence output, which is the gap our head fills.

- p. 8, §7.1, ¶1, line 1: "we use DPT [55] as our depth decoder, built on DINOv2 encoders" (the decoder our `output_conv1` tap sits in)
- p. 8, §7.1, ¶1, lines 1-2: trained at 518×518, shorter side resized to 518 (our input is 924 × 518)
- [main.tex:151](../latex/main.tex:151): backbone paragraph; frozen relative ViT-B, output is disparity with unknown scale and shift, two per-image least-squares fits (lidar and radar)
- [main.tex:199](../latex/main.tex:199): deviations table row D1; proposal "Metric checkpoint (vkitti), in meters, no fit" versus implemented "Relative ViT-B with per-image lidar and radar fits", status "Decided; lidar leakage noted"
- Absence claim, not quotable: the words "uncertainty" and "confidence" do not appear as a model output anywhere in pp. 1-30. The only related step is training-time: p. 6, §5.2, ¶1, lines 2-4 ignore the top 10% largest-loss regions of each pseudo-labeled sample as likely noisy.

### notes for the writer

- The paper does not say why one would choose the relative checkpoint over the metric one. The reason for D1 has to come from our own docs, not from this citation.
- `references.bib` lists this as an arXiv preprint; the PDF footer on p. 1 says "38th Conference on Neural Information Processing Systems (NeurIPS 2024)".
- Table 2's caption (p. 8) says V2 is only comparable to V1 on standard benchmarks and that its gains (fine detail, transparent objects) do not show there. Do not claim V2 beats V1 on KITTI: ViT-B is 0.078 versus 0.080 AbsRel and identical δ1.

## post-review discussion with writer

### Q1: if V2 is 10× faster than diffusion-based models, what kind of model is V2, and V1? what is the underlying architecture of the Depth Anything models?

- A1: Both are discriminative, feed-forward regressors: a DINOv2-pretrained Vision Transformer encoder followed by a DPT convolutional decoder, mapping an image to a depth map in one forward pass. Diffusion-based models such as Marigold are generative and run a denoising network many times per image, which is where the speed gap comes from. V1 and V2 share this architecture; V2 changes the training data and the teacher size, not the network.
  - V2 p. 2, §1, ¶2, lines 1-3: the two groups, "discriminative models" (BEiT, DINOv2) versus "generative models" (Stable Diffusion)
  - V2 p. 2, §1, ¶4 (after the bullets), line 1: "the nature of MDE is a discriminative task", so they start from V1
  - V2 p. 8, §7.1, ¶1, line 1: "we use DPT [55] as our depth decoder, built on DINOv2 encoders"
  - V1 ([2401.10891](../../Papers/2401.10891.pdf)) p. 5, §4.1, ¶1, lines 1-2: "We adopt the DINOv2 encoder" and the DPT decoder, following MiDaS
  - V2 p. 1, Figure 1: latency and parameter counts against Marigold and DepthFM

### Q2: what is affine-invariant depth, how is it related to the scale- and shift-invariant and gradient matching losses, and how does it relate to our use of inverse lidar depth?

- A2: Affine-invariant inverse depth means the output d is only correct up to an unknown per-image scale s and shift t: true inverse depth (1/z) ≈ s·d + t. That is a direct consequence of the scale- and shift-invariant loss, which normalises away scale and shift in prediction and label before comparing them, so the model is never penalised for getting them wrong and never learns them; the gradient matching loss is computed on the same aligned maps and only sharpens edges. We recover s and t ourselves, per image, by least squares against inverse lidar depth (and separately against radar), and the fit has to be done in inverse depth because that is the only space where the model's output is linear in the truth.
  - V2 p. 6, §5.2, ¶1, lines 4-7: "affine-invariant inverse depth", then the two loss terms Lssi and Lgm, credited to MiDaS
  - V2 p. 6, §5.2, ¶1, lines 7-8: Lgm is "super beneficial to the depth sharpness" with synthetic images
  - MiDaS ([1907.01341](../../Papers/1907.01341.pdf)) p. 5, paragraph "Scale- and shift-invariant losses", lines 1-2: prediction in "disparity space (inverse depth up to scale and shift)"
  - MiDaS p. 5, Eq. 2 and the text around it: scale and shift aligned by "a least-squares criterion", solved in closed form
  - MiDaS p. 6, paragraph "Final loss", lines 1-3 and Eq. 11-12: the gradient matching term, which "biases discontinuities to be sharp"
  - [main.tex:151](../latex/main.tex:151): our two per-image least-squares fits of disparity to inverse depth (lidar for labels and the temporal channel, radar for the radar residual)

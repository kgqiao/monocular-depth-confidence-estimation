# literature review evidence: [Vision Transformers for Dense Prediction (DPT)](../../Papers/2103.13413.pdf)

Ranftl, Bochkovskiy and Koltun, ICCV 2021 (the local PDF is the arXiv v1 preprint) · bib key `ranftl2021dpt` · report §2.1

## ai-generated summary review

### what the paper does

It introduces the dense prediction transformer (DPT): a vision transformer (ViT) encoder whose tokens, taken from four transformer stages, are reassembled into image-like feature maps at four resolutions and then merged by a convolutional decoder into a dense prediction. Trained on a 1.4M-image depth meta-dataset, it improves zero-shot monocular depth by up to 28% relative to the fully-convolutional MiDaS network.

- p. 1, Abstract, lines 1-7: ViT "in place of convolutional networks as a backbone", tokens assembled into "image-like representations at various resolutions" and combined "using a convolutional decoder"
- p. 3, right column, paragraph "Convolutional decoder", lines 1-7: the "three-stage Reassemble operation" that turns tokens from arbitrary encoder layers into feature maps
- p. 4, left column, paragraph starting "Irrespective of the exact transformer backbone", lines 1-8: features reassembled "at four different stages and four different resolutions"; ViT-Base taps layers "l = {3, 6, 9, 12}"
- p. 1, Abstract, lines 15-18: "an improvement of up to 28%" in relative performance for monocular depth
- p. 5, Table 1 and left column, ¶1, lines 1-2 (top of column): DPT-Large and DPT-Hybrid versus MiDaS on six zero-shot datasets; the MIX 6 training set "contains about 1.4 million images"

### the one result or idea that matters to us

The decoder is convolutional and built from RefineNet-based feature fusion blocks: each block combines two feature maps with residual convolutional units and upsamples by two, and after the last block a three-convolution depth head finishes the prediction. The first convolution of that head halves the feature dimension, the result is upsampled, and the last two convolutions (output dimension 32, then a scalar) produce inverse depth.

- p. 4, last paragraph of left column continuing at top of right column, lines 1-5: "using a RefineNet-based feature fusion" block (the phrase breaks across the column) and upsampling "by a factor of two in each fusion stage"
- p. 4, right column, same paragraph, lines 5-7: final representation "has half the resolution of the input image"; "We attach a task-specific output head"
- p. 3, Figure 1 (left, right) and caption lines 5-7: four "Fusion" modules, one per "Reassemble" output, feeding a "Head"; the right panel shows a fusion block as two "Residual Conv Unit" boxes, an addition, "Resample0.5" and "Project"; caption: "Fusion blocks combine features using residual convolutional units [23]"
- p. 5, left column, ¶2 (starts "We use multi-objective optimization"), lines 5-9: "an output head that consists of 3 convolutional layers", which halves the feature dimension and upsamples "after the first convolutional layer"
- p. 11, Supplementary §A, left column, paragraph "Monocular depth estimation head", lines 3-8: "The initial convolution halves the feature dimensions", second convolution "has an output dimension of 32", final layer gives a non-negative scalar for inverse depth; p. 12, Figure A1 (b): Conv3x3, Resample, Conv3x3-ReLU, Conv1x1-ReLU

### how it relates to our design

Depth Anything V2 uses this decoder, and the 64 feature channels our head reads come from it. In the vendored Depth Anything V2 code the four fusion blocks are named `refinenet4` to `refinenet1` and `output_conv1` is the first convolution of the depth head, applied to the output of the last fusion block (`refinenet1`) and halving its channels (128 to 64 for ViT-B); this matches the paper's "initial convolution" of the depth head, but the names themselves are from the code, not the paper.

- p. 11, Supplementary §A, left column, paragraph "Monocular depth estimation head", lines 3-4: "The initial convolution halves the feature dimensions" (the layer the code calls `output_conv1`)
- [dpt.py:101-104](../../../confhead/src/confhead/depth_anything_v2/dpt.py:101): `refinenet1` to `refinenet4` created as fusion blocks; [dpt.py:109](../../../confhead/src/confhead/depth_anything_v2/dpt.py:109): `output_conv1` defined as a 3×3 convolution from `features` to `features // 2`; [dpt.py:145-149](../../../confhead/src/confhead/depth_anything_v2/dpt.py:145): `refinenet1` output goes into `output_conv1`, then bilinear upsampling, then `output_conv2`
- [depth.py:6-10](../../../confhead/src/confhead/depth.py:6): our own description of the tap, "64 channels at 8x, the last decoder features before the depth regression convolutions"; [depth.py:34](../../../confhead/src/confhead/depth.py:34): ViT-B uses `features` 128
- [progress.tex:23](../latex/sections/progress.tex:23): input table row, channels 0-63 are "Decoder features", "`output_conv1` tap, 64 channels, area-averaged to the grid"
- [literature-review.tex:19](../latex/sections/literature-review.tex:19) and [literature-review.tex:23](../latex/sections/literature-review.tex:23): the §2.1 framing sentence and the red stub for this paper ("convolutional RefineNet-style decoder; name the fusion blocks and the `output_conv1` tap we read")
- [progress.tex:10](../latex/sections/progress.tex:10): the backbone is the frozen relative Depth Anything V2 ViT-B; p. 5, right column, paragraph "Fine-tuning on small datasets", lines 4-5: DPT itself was "trained with an affine-invariant loss", so its predictions "are arbitrarily scaled and shifted" (the same output convention we fit per image)

### notes for the writer

- Absence claim, not quotable: the strings `output_conv1`, `refinenet1` to `refinenet4` and `scratch` do not appear anywhere in pp. 1-15. The paper's names are "Fusion" module or block, "RefineNet-based feature fusion block", "Residual Conv Unit", "Reassemble" and "output head". If the report names `refinenet1` or `output_conv1`, attribute the names to the Depth Anything V2 code (dpt.py lines above) and cite the paper only for the structure. The paper says "RefineNet-based", not "RefineNet-style".
- [literature-review.tex:19](../latex/sections/literature-review.tex:19) says "the decoder is a dense-prediction transformer whose intermediate feature maps are easy to tap". In the paper DPT is the whole encoder-decoder architecture, the transformer is the encoder, and the decoder is convolutional (p. 2, left column, ¶2, lines 2-11: "leverages a transformer as the basic computational building block of the encoder"). "Easy to tap" is not a claim the paper makes; its only use of "tap" is about which encoder layers feed the decoder (p. 7, paragraph "Skip connections").
- Numbers in this paper are not our backbone's: DPT here uses ImageNet-pretrained ViT with patch size 16 and 256-dimensional decoder feature maps (p. 3, left column, paragraph starting "The input tokens are transformed", lines 10-11: "patch size p = 16"; p. 4, left column, paragraph starting "Irrespective of the exact transformer backbone", lines 11-12: "D̂ = 256"), while our ViT-B Depth Anything V2 uses 128 (depth.py:34). The halving and the 32-channel second convolution carry over in the code; the 256 does not. Also, the local PDF is arXiv v1 (24 Mar 2021) and carries no ICCV footer, so the ICCV 2021 venue in `references.bib` cannot be checked against it; title and authors match.

## post-review discussion with writer

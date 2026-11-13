# literature review evidence: [HyDRa](../../Papers/2403.07746v4.pdf)

Wolters et al., arXiv preprint arXiv:2403.07746v4 (2025) · bib key `wolters2025hydra` · report §2.3

## ai-generated summary review

### what the paper does

HyDRa is a camera-radar architecture for 3D detection, tracking and semantic occupancy that fuses radar with image features in two places: in the perspective view, where a Height Association Transformer (HAT) feeds radar pillar features into the image features before the depth network, and in BEV, where a Radar-weighted Depth Consistency (RDC) weights the backward-projection attention. It reports 64.2 NDS and 58.4 AMOTA on the nuScenes test set (state of the art for camera-radar fusion) and 44.4 mIoU on Occ3D, 3.7 above the best camera-only method.

- p. 1, Abstract, lines 12-16: HAT uses "radar features already in the perspective view" for depth; in BEV the sparse representation is refined "by a Radar-weighted Depth Consistency"
- p. 1, Abstract, lines 16-22: "radar fusion of 64.2 NDS (+1.8) and 58.4 AMOTA (+1.5)" on nuScenes; beats "all previous camera-based methods on the Occ3D benchmark" by 3.7 mIoU
- p. 3, §III, bullet 2 "Unified Depth Prediction" (left column), lines 1-7: cross-attention to "associate pillar features (missing height, sparse depth)" with image columns; result is added residually and converted into a dense depth distribution
- p. 4, §III-C, ¶1 as continued from p. 3 (left column, under Table II), lines 5-9: a radar guidance network (3×3 convolution and sigmoid) encodes "the radar-only BEV features to additional attention weights"; then p. 4, Eq. 3 and the 4 lines after it (right column): deformable attention "weighted by depth consistency" w_c and radar attention r
- p. 4, Table II (left column), row HyDRa: test NDS 64.2, mAP 57.4, mATE 0.398; p. 5, Table V (right column), row HyDRa: mIoU 44.4 against FB-OCC 40.7

### the one result or idea that matters to us

The paper names the same radar problem we measure: nuScenes-style radar has no elevation and is sparse and noisy, so tying a radar return to the right image pixel is the hard part. Its answer is to learn the association along the whole image column instead of trusting the projected point, and in the front-view ablation adding HAT to BEVDepth takes NDS from 44.4 to 50.3 and translation error (mATE) from 0.737 to 0.586, better than a baseline that projects radar into the image and runs 2D convolutions on it (48.1 NDS).

- p. 1, §I, ¶4 (right column), lines 3-7: "The absence of elevation information", sparse returns, multi-path noise and high measurement uncertainty make radar hard to associate with camera features
- p. 3, §III-B, ¶1 (right column), lines 5-10: the "primary challenge" is associating radar and image features in the perspective view; the module gets a receptive field "along the complete image height"
- p. 3, §III-B, ¶2 (right column), last 3 lines: the fused map is the input "for the depth and context network", giving the final depth distribution (this is the "before the depth head" claim)
- p. 6, Table VI (left column), rows BEVDepth / BEVDepth + RF / BEVDepth + HAT / HyDRa (RDC + HAT): NDS 44.4 / 48.1 / 50.3 / 53.6, mATE 0.737 / 0.614 / 0.586 / 0.493; caption says trained and evaluated "in the front-view setting"
- p. 6, §IV-D, "Effect of the Range View Fusion", ¶2 (left column), lines 5-7: processing the "sparse stripe of radar points with 2D convolutions" is "neither effective nor efficient"

### how it relates to our design

HyDRa puts radar inside the network so that the predicted depth distribution itself changes, and it trains that network for its 3D tasks; we leave the depth model frozen and give radar to a separate head only as a residual channel and a validity mask, so the output is a confidence map and the depth is unchanged. Our radar channels are built by projecting radar points onto the image grid, which is the kind of input the paper's Range-Fusion baseline uses and which HAT is designed to improve on, and our report attributes its noisy radar residual to the same missing elevation the paper describes.

- p. 2, Fig. 2 caption, line 2: radar is associated with image features by HAT and "the resulting radar-informed dense depth" drives the forward projection (radar changes the depth)
- p. 5, §IV-B, ¶2 (left column), lines 1-2: built on the BEVDet code base, "train for 20 epochs" (the model is trained for the task; the paper does not describe a frozen depth model. The second half is an absence claim, not quotable)
- p. 6, §IV-D, "Effect of the Range View Fusion", ¶1 (left column), lines 6-9: the RF baseline is made by "projecting radar points into the image plane", with every non-radar pixel set to zero
- [main.tex:113](../latex/main.tex:113): our stance; radar channels tell the head where the frozen depth disagrees with an independent range measurement, "so the output is a confidence map rather than a better depth map"
- [main.tex:166](../latex/main.tex:166) and [main.tex:167](../latex/main.tex:167): the two radar inputs, residual |d̂ − z| / z under the radar fit (5 sweeps, radius 1 cell) and a validity mask; [main.tex:230](../latex/main.tex:230): radar residual about five times the lidar error, likely cause "nuScenes radar has no elevation"; [main.tex:140](../latex/main.tex:140): the contrast sentence for the section summary
- Absence claim, not quotable: the paper has no confidence or uncertainty output and reports no per-pixel depth error (no AbsRel, RMSE or similar) anywhere in pp. 1-10. "uncertainty" appears once, for radar measurement noise (p. 1, §I, ¶4, right column, lines 5-6).

### notes for the writer

- Scope the "state of the art" wording in the stub at [main.tex:117](../latex/main.tex:117). On nuScenes the claim is state of the art for camera-radar fusion (the lidar CenterPoint-V row in Table I, p. 4, has 65.3 NDS against HyDRa's 61.7 on val, and CRN has a higher test mAP, 57.5 against 57.4, in Table II). On Occ3D the claim is against camera-based methods only (Abstract lines 20-22; p. 6, §V, right column, lines 4-8), with HyDRa called "the first radar-enhanced occupancy prediction model". The margin over CRN on the test set is 1.8 NDS in the abstract, the conclusion and Table II (64.2 against 62.4); p. 5, §IV-C, ¶1, line 13 says "by 1.4 NDS", so use 1.8.
- "More accurate depth" is argued indirectly. The paper never measures depth error; it attributes the lower translation error to depth (p. 5, §IV-C, ¶1, left column, lines 8-11: mATE of 0.416 "can be attributed to our model's robust depth estimation concept"). The depth is also a per-pixel distribution over depth bins used for BEV pooling (118 categories at the smallest setting, p. 5, §IV-B, ¶1, line 3), not a depth map like Depth Anything's. "They change the depth, we score it" is safe; "they improve depth accuracy by X" is not.
- `references.bib` lists this as an arXiv preprint, year 2025, and that agrees with the PDF: the only publication mark is the arXiv stamp "arXiv:2403.07746v4 [cs.CV] 26 Mar 2025" on p. 1, with no venue. Author list and title match. The paper abbreviates the BEV component as RDC (p. 2, contribution 2) and RWDC (p. 1, Fig. 1); spell the name out. Optional extra for the adverse-conditions thread: Appendix A, Table VIII (p. 9) reports Occ3D mIoU at night of 30.1 against 25.8 for camera-only FB-OCC (+16.7%), an occupancy result, not a depth one.

## post-review discussion with writer

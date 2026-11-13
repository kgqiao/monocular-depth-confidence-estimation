# literature review evidence: [Depth Estimation from Monocular Images and Sparse Radar Data](../../Papers/2010.00058.pdf)

Lin, Dai, and Van Gool, IROS 2020 · bib key `lin2020radardepth` · report §2.3

## ai-generated summary review

### what the paper does

It trains a ResNet-18 encoder-decoder on nuScenes that takes an RGB image plus a sparse projected radar depth map and predicts dense depth, with projected lidar as ground truth. The network has two stages: stage 1 predicts a coarse depth, radar points that disagree with it by more than a depth-dependent tolerance are discarded, and stage 2 predicts the final depth from the image, the filtered radar, and the stage-1 depth.

- p. 1, Abstract, lines 1-3: the goal, "fusing monocular" images and radar points with a deep network (line 2)
- p. 3, §III-A, paragraph "Problem formulation", left column, lines 3-8: radar "from 3 nearest timestamps" projected to a single-channel depth map; projected lidar "is treated as ground truth"
- p. 3, §III-B, paragraph "Two-Stage architecture" through Eq. 2, right column: stage 1 gives a coarse depth, then "to reject some outliers" (paragraph after Eq. 1, line 2), then stage 2 takes image, filtered radar and stage-1 depth
- p. 4, §III-C, left column, Eq. 3 and Eq. 4: the tolerance τ(d) grows with depth ("larger tolerance for large depth values", ¶2, line 4) and a radar point is kept only if |radar depth − stage-1 depth| ≤ τ
- p. 6, Table III, rows "RGB only" and "Ours two-stage w/ smoothness": δ1 0.862 → 0.901, RMSE 5.613 → 5.18, REL 0.126 → 0.1

### the one result or idea that matters to us

The paper names four limitations of nuScenes radar relative to lidar: sparseness, a limited vertical field of view (returns sit at similar heights), noisy measurements from multipath, and inconsistency with lidar even for noise-free returns. It concludes that this noise is the main reason lidar-fusion methods fail on radar: fed raw radar, most of them do not beat the RGB-only baseline.

- p. 3, §III-A, paragraph "Limitations", left column, bullet 2, lines 1-4: "Limited vertical field of view"; radar measurements mainly concentrate "in the central horizontal bin (similar heights)" (line 3), illustrated in Fig. 2 (right)
- p. 3, §III-A, "Limitations", bullet 3, lines 1-3: "Noisy measurements: Due to multiple reflections (Radar" multipath problem) "or other reasons", illustrated in Fig. 2 (left)
- p. 3, §III-A, "Limitations", bullet 4, lines 1-7: noisy measurements are "considered as outliers"; radar and lidar points on the same object can differ, so with lidar as ground truth "even noise-free" radar is not perfect on the metrics (line 5)
- p. 3, §III-A, "Limitations", bullet 1, lines 1-4: more than 3000 lidar points but "less than 100 Radar points" after projection to the camera
- p. 6, §IV-D, right column, paragraph "The results are presented in Table III", lines 2-8: with raw radar "most of the method cannot yield improvements over the" RGB-only baseline; the "denoising stage is crucial" (line 7). Also p. 1, Abstract, lines 6-10: the noise is "one of the main" key reasons existing lidar fusion methods do not transfer

### how it relates to our design

Their filter and our radar residual both compare a projected radar range with a predicted depth at the same pixel; they threshold the difference to drop radar points before a second depth network, while we pass the relative difference to the head as an input channel and leave the depth model frozen. Both accumulate radar over time to raise the point count (3 nearest timestamps, about 40-100 points per image for them; 5 sweeps, 150 points per image on mini for us), and the noise they describe qualitatively is what we measure as a median radar residual of 0.25.

- p. 4, §III-C, left column, Eq. 4: keep point p_n if |x_Radar(p_n) − ỹ_stage1(p_n)| ≤ τ(p_n); p. 3, §III-B, paragraph after Eq. 1, right column, lines 6-8: stage-1 depth is suitable to reject "outlier noises produced by Radar multipath"
- p. 4, §IV-A, right column, bullet 3 ("Radar depth map"), lines 3-6: "we will accumulate Radar points" from 3 nearest timestamps "to increase the point counts", "(Around 40-100 points / image)"
- [main.tex:166](../latex/main.tex:166): our channel 66, radar residual |d̂ − z| / z under the radar fit, 5 sweeps, radius 1 cell; [main.tex:167](../latex/main.tex:167): the validity mask
- [main.tex:113](../latex/main.tex:113): our stated contrast, radar channels tell the head where the frozen depth disagrees with an independent range measurement, "a confidence map rather than a better depth map"
- [main.tex:230](../latex/main.tex:230): radar residual median 0.25 (radar fit) and 0.28 (lidar fit), about five times the lidar-measured error; [main.tex:244](../latex/main.tex:244): 150 radar points per image at 5 sweeps

### notes for the writer

- The paper does not say nuScenes radar has "no elevation", and the word "elevation" does not appear anywhere in pp. 1-9 (Absence claim, not quotable). What it says is weaker: a "limited vertical field of view" with returns at "similar heights" (p. 3, bullet 2; p. 1, §I, ¶3, right column, lines 10-12: "typically sparser, noisier, and have a more limited vertical" field of view). It also lists z among the radar fields (p. 2-3, §III-A, paragraph "Data format": "in addition to x, y, z"). Cite it for "radar returns are confined to a narrow height band and are noisy", not for "no elevation"; the stub at [main.tex:115](../latex/main.tex:115) and the HyDRa sentence at [main.tex:117](../latex/main.tex:117) both use the stronger wording.
- The paper gives no number for radar error against lidar, so "about five times the lidar error" is our measurement only (Absence claim, not quotable). Its evidence for noise is Fig. 2 and the tables, and the tables are mixed: in Table II (p. 5) radar filtered with lidar ground truth is almost identical to raw radar under early fusion (δ1 0.877 vs 0.876, REL 0.115 both), so the support for filtering is Table III (p. 6): REL 0.112 single-stage, 0.102 and 0.1 two-stage.
- The PDF is the arXiv version (v1, 30 Sep 2020) and carries no venue line, so IROS 2020 in `references.bib` cannot be checked against it. The lidar-based radar filter used in the pilot experiment is not described (p. 6, footnote 3), and K in Eq. 3 is not defined in the text, so do not restate the threshold formula in detail.

## post-review discussion with writer

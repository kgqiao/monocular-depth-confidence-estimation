# literature review evidence: [md4all: Robust Monocular Depth Estimation under Challenging Conditions](../../Papers/2308.09711.pdf)

Gasperini et al., ICCV 2023 · bib key `gasperini2023md4all` · report §2.4

## ai-generated summary review

### what the paper does

It makes a standard monocular depth network robust at night and in rain by changing only the training: day-clear images are translated to night and rain with a GAN, the network is fed a mix of original and translated images, and the loss is always computed against the original day-clear sample (its photometric loss, its lidar, or the output of a frozen day-only model). The architecture and inference are unchanged, and the method is shown for a self-supervised model (Monodepth2-based) and a lidar-supervised one (AdaBins) on nuScenes and Oxford RobotCar.

- p. 1, Abstract, left column, lines 11-18: generate hard samples, then feed them while "computing the standard losses on the correspond-" ing original images; "without mod-" ifications at inference time
- p. 3, §3, ¶1, right column, lines 6-9: "adverse image translation, train on the generated adverse" samples, learning only from the original day inputs
- p. 5, §3.1.3, ¶2, left column, lines 3-5 and Eq. 1: md4all-DD distils from "a depth network B (baseline) trained at an earlier" stage on easy samples only
- p. 6, §4.1, paragraph "Image translation", left column, lines 6-8: one ForkGAN per condition on nuScenes, "C = {night, rain}"
- p. 6, §4.1, paragraph "Prior works and baselines", right column, lines 3-4: applied to "the self-supervised Monodepth2-based baseline" and to the fully-supervised AdaBins

### the one result or idea that matters to us

The paper fixes a per-condition evaluation of monocular depth on the nuScenes validation set: 6019 validation images, "of which 4449 day-clear, 602 night, and 1088 rain", scored against a single lidar scan up to 80 m, and it shows depth error roughly doubling from day-clear to night for most self-supervised methods. It also states that nuScenes has no weather or illumination metadata for the test split, so its test-set numbers are for all conditions combined.

- p. 5, §4.1, paragraph "Datasets and metrics", right column, line 6, continuing on p. 6, left column, lines 1-4: the three groups, "night (including night-rain), and day-rain", then "(with synced sensors), and 6019 validation ones (of which" 4449 day-clear, 602 night, and 1088 rain)
- p. 6, §4.1, same paragraph, left column, lines 12-13 on the page: errors up to "80m for nuScenes as in [9]"; p. 11, §A.2.2, ¶1, left column, lines 2-4: predictions compared with a single lidar scan, "gle LiDAR scan. As in R4Dyn, we discarded static frames"
- p. 6, §4.2, paragraph "Night – nuScenes", right column, lines 4-6: "rors of most methods to double from ideal conditions"; Table 1 (p. 6), row Monodepth2 trained on d: absRel 0.1374 day-clear, 0.2828 night, 0.1727 day-rain
- p. 6, Table 1, row md4all-DD: absRel 0.1366 day-clear, 0.1921 night, 0.1414 day-rain; §4.2, same paragraph, lines 20-21: at night it "absRel by 32% and relatively increased δ1 by 37% (DD)" relative to Monodepth2
- p. 18, §A.3.6, ¶1, left column, lines 2-4: "Since no metadata about the weather and" illumination conditions is available for the test set; p. 18, Table 14 caption, line 4: "nuScenes provides no condition annotations for the test set"

### how it relates to our design

md4all retrains the depth network so that it fails less at night and in rain; we leave the depth network frozen and train a head to flag where it fails, and we report results for the same three condition groups on the nuScenes validation set. The paper also warns that lidar, our label source, returns erroneous measurements in rain, so a lidar-scored error in rain is not fully trustworthy.

- p. 2, §1, paragraph starting "In adverse weather conditions such as rain", left column, lines 3-6: "radars are robust in such conditions, LiDARs become unre-" liable, with multi-path and blooming effects
- p. 7, §4.2, paragraph "Fully-supervised", left column, lines 6-9: "that LiDAR artifacts may have an impact on the rain val-" ues, so that perfect estimates would not score perfectly
- p. 7, §4.2, paragraph "Rain – nuScenes", left column, lines 1-3: "Rain is less problematic than darkness"; all methods do better in rain than at night
- [introduction.tex:7](../latex/sections/introduction.tex:7): motivation; camera depth is least reliable in "night, rain on the lens, spray, and low sun", and a confidence map lets a consumer weight or discard the depth
- [introduction.tex:9](../latex/sections/introduction.tex:9): the head predicts the error "without touching the depth model"; lidar projected into the image gives the true error at labeled cells
- [results.tex:13](../latex/sections/results.tex:13): our own cache shows the same direction as the paper's Table 1, median per-image AbsRel 0.052 by day and 0.079 at night (mini split)
- [plan.tex:16](../latex/sections/plan.tex:16): planned per-condition split, 111 day, 24 rain, 15 night val scenes with 4,449, 968, and 602 keyframes, attributed to BEVCar; [progress.tex:69](../latex/sections/progress.tex:69): D11, how to count "night, after rain" scenes is still open
- Absence claim, not quotable: md4all outputs depth only. No confidence or uncertainty output is described anywhere in pp. 1-30; the only occurrence of "uncertainty" in the body is a citation of Poggi et al. in related work (p. 3, §2.2, ¶1, left column, last two lines, "uncertainty estimation [27]").

### notes for the writer

- The rain count does not match ours and the paper's three counts do not add up: 4449 + 602 + 1088 = 6139, but the paper gives 6019 validation images. [plan.tex:16](../latex/sections/plan.tex:16) uses 968 rain keyframes (4,449 + 968 + 602 = 6,019). The gap is 120 in both cases, which would fit 120 night-rain images being counted under both "night" and "rain", but the paper does not say this (it says night includes night-rain and names the third group "day-rain" in one sentence and "rain" in the next). So cite md4all for 4,449 and 602, and either cite 1,088 as "rain" with this caveat or keep 968 with BEVCar as the source. Also, the paper says "images", not "keyframes", and does not name the camera in that sentence.
- The paper does not define the weather split by scene lists or a rule; it says only that it "used the of-" ficial split following R4Dyn (p. 6, left column, lines 1-2). How a scene is assigned to night or rain is not in this paper.
- The PDF in `Docs/Papers` is the arXiv version and prints no venue, so "ICCV 2023" in `references.bib` cannot be checked against it (Absence claim, not quotable). The PDF title is "Robust Monocular Depth Estimation under Challenging Conditions"; "md4all" is the method name, matching the bib title.

## post-review discussion with writer

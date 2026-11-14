# literature review evidence: [BEVCar](../../Papers/2403.11761.pdf)

Schramm et al., IROS 2024 (the PDF is arXiv v2 and carries no venue mark) · bib key `schramm2024bevcar` · report §2.3

## ai-generated summary review

### what the paper does

BEVCar fuses six surround-view cameras with automotive radar to segment vehicles and seven map classes in the bird's-eye view on nuScenes. It learns a point-based radar encoding, uses the radar points to initialize the queries that lift image features into BEV, and reaches 58.4 vehicle IoU against 48.8 for its own camera-only variant.

- p. 1, Abstract, lines 10-15: "a novel approach for joint BEV object and map segmentation"; the novelty is a learned radar encoding that initializes the image lifting
- p. 3, §III-A, paragraph "Camera", lines 1-2 (left column): "we employ a frozen DINOv2 ViT-B/14 [26]", with a learnable ViT adapter
- p. 3, §III-A, paragraph "Radar", lines 5-8 (right column): only position, uncompensated velocities and RCS per point are used; "we propose to learn a radar representation" (lines 9-10)
- p. 5, §IV-A, paragraph "Dataset and Metrics", lines 12-16 (left column): vehicle subclasses merged into one class; the seven map classes, starting at "we consider all available classes"
- p. 5, Table I, rows "BEVCar (camera)" and "BEVCar (ours)": vehicle 48.8 vs 58.4, map 50.5 vs 54.3, mIoU 65.0 vs 70.9 on the nuScenes validation split

### the one result or idea that matters to us

The paper divides the nuScenes validation set into day (4,449 samples), rain (968 samples) and night (602 samples) and releases that split with its code. On it, the camera-only variant loses vehicle IoU at night (48.7 day, 42.1 night) while the camera-radar model holds (58.3 day, 59.1 rain, 57.4 night).

- p. 5, §IV-A, paragraph "Dataset and Metrics", lines 7-9 (left column): official split, "containing 28,130 and 6,019 samples, respectively"
- p. 5, §IV-A, paragraph "Dataset and Metrics", lines 9-12 (left column): "categorize the validation scenes into day (4,449 samples), rain" (968 samples) and night (602 samples), "include this split in our code release"
- p. 2, §I, contribution 5 (left column, top): "We make the used day/night/rain splits on nuScenes [2] publicly available"
- p. 6, Table IV (right column), rows "BEVCar (camera)" and "BEVCar (ours)": vehicle IoU 48.7 / 49.9 / 42.1 and 58.3 / 59.1 / 57.4 for day / rain / night; map mIoU 53.3 / 45.1 / 39.2 and 57.3 / 48.8 / 42.4
- p. 7, §IV-C, paragraph "Robustness to Weather and Illumination", second paragraph (begins "In Tab. IV, Fig. 1, and Fig. 5"), lines 4-7 (left column): camera-only vehicle IoU "is subject to substantial degradation during the night"; camera-radar methods maintain it

### how it relates to our design

We plan to report every result per BEVCar condition group and to adopt BEVCar's released scene lists and its rule for rainy night scenes, replacing our current keyword rule that counts such a scene as both night and rain. BEVCar uses radar as an input that improves a BEV segmentation output; our report uses radar as an input to a head that predicts where a frozen depth model is wrong.

- p. 5, §IV-A, paragraph "Dataset and Metrics", lines 9-12 (left column): the split and its release, "categorize the validation scenes into day (4,449 samples), rain"
- p. 5, §IV-A, paragraph "Implementation Details", lines 11-12 (right column): "we aggregate five radar sweeps as input" (our radar residual also uses 5 sweeps, [progress.tex:26](../latex/sections/progress.tex:26))
- [literature-review.tex:55](../latex/sections/literature-review.tex:55): the stub for this paper; "the source of the day/rain/night split of the nuScenes val scenes (111/24/15 scenes)"
- [plan.tex:16](../latex/sections/plan.tex:16): plan item; adopt "the 111 day, 24 rain, and 15 night val scenes (4,449, 968, and 602 keyframes) from BEVCar's released lists"
- [progress.tex:69](../latex/sections/progress.tex:69) and [plan.tex:21](../latex/sections/plan.tex:21): deviation D11; current substring rule makes a "night, after rain" scene both night and rain; planned resolution "adopt the BEVCar rule for rainy night scenes"
- [results.tex:57](../latex/sections/results.tex:57) and [results.tex:77](../latex/sections/results.tex:77): results are scored "per BEVCar condition group"; [literature-review.tex:47](../latex/sections/literature-review.tex:47): we use radar for a confidence map "rather than a better depth map"

### notes for the writer

- The scene counts 111/24/15 are not in the paper. It gives only sample counts (4,449 / 968 / 602) and never says how a rainy night scene is assigned. Absence claim, not quotable: the words "111", "after rain" and any assignment rule do not appear on pp. 1-8. Both facts must be taken from, and cited to, the released split files at http://bevcar.cs.uni-freiburg.de (p. 1, Abstract, lines 20-23). What the paper does support: 4,449 + 968 + 602 = 6,019, the full validation set (p. 5, same paragraph, lines 9-12), so the three groups do not overlap and each rainy night scene sits in exactly one group. That is our arithmetic, not a statement in the paper, and it does not say which group.
- BEVCar's rain count (968) differs from the md4all count in the stub at [literature-review.tex:65](../latex/sections/literature-review.tex:65) (1,088); day and night agree. The paper does not mention md4all or explain the difference, so do not present the two splits as the same.
- `references.bib` gives IROS 2024, pp. 1435-1442. The PDF is the arXiv version ("arXiv:2403.11761v2 [cs.RO] 25 Jul 2024", p. 1 margin) and names no venue or page range, so the venue cannot be verified from this file. Also, the paper calls the 6,019 units "samples", not keyframes, and its radar benefit is shown for vehicle segmentation; map mIoU still falls in rain and at night for every method (p. 7, same paragraph, lines 8-10: "decreases during rain and even further at night").

## post-review discussion with writer

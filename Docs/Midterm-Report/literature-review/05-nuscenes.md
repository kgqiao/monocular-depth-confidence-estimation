# literature review evidence: [nuScenes: A Multimodal Dataset for Autonomous Driving](../../Papers/1903.11027.pdf)

Caesar et al., CVPR 2020 · bib key `caesar2020nuscenes` · report §2.4

## ai-generated summary review

### what the paper does

It releases a driving dataset of 1000 scenes of 20 s each, recorded in Boston and Singapore with 6 cameras, 1 lidar (32 beams) and 5 radars covering 360°, annotated with 3D boxes for 23 classes at 2 Hz. It also defines 3D detection and tracking metrics and gives lidar and image baselines; it is not a depth-estimation paper.

- p. 1, Abstract, lines 12-16: "6 cameras, 5 radars and 1 lidar", then "nuScenes comprises 1000 scenes", each 20s long, 3D boxes for 23 classes and 8 attributes
- p. 3, §2, paragraph "Drive planning", right column, lines 1-3: "We drive in Boston (Seaport and South" Boston) and Singapore (One North, Holland Village, Queenstown)
- p. 4, Table 2 (top of left column), rows "6x Camera", "1x Lidar", "5x Radar": cameras 12Hz at 1600 × 900; lidar "Spinning, 32 beams", 20Hz, ≤ 70m range; radar "≤ 250m range, 77GHz, FMCW", 13Hz
- p. 4, §2, paragraph "Scene selection", right column, lines 1-3: "manually select 1000 interesting scenes of 20s duration"
- p. 4, §2, paragraph "Data annotation", right column, lines 1-2: "keyframes (image, lidar, radar) at 2Hz"; lines 9-13: intermediate (non-keyframe) sensor frames are also released, at 12Hz, 13Hz and 20Hz for camera, radar and lidar

### the one result or idea that matters to us

It puts camera, lidar and radar in the same recordings together with night and rain data, and claims to be the first AV dataset to include radar and the first multimodal one with night and rain. The paper itself describes radar as longer-range than lidar and able to measure velocity, but sparser and less precise in localization.

- p. 2, §1.1, ¶2, right column, lines 1-5 of the column (the paragraph starts at the bottom of the left column): "first AV dataset to include radar data"; first multimodal dataset with "nighttime and rainy conditions"
- p. 1, §1, ¶2, right column, lines 11-14: radar reaches 200-300m and measures velocity through the Doppler effect, but "the returns are even sparser than lidar" and "less precise in terms of localization"
- p. 5, §2, paragraph "Annotation statistics", left column, ¶1, lines 6-10: 40k keyframes; Boston 55%, the three Singapore areas 21.5%, 13.5%, 10%; "rain: 19.4%" and "night: 11.6%"
- p. 5, §2, paragraph "Annotation statistics", left column, ¶2, lines 8-12: an annotated object holds up to 100 lidar points even at 80m, but only "up to 40 radar returns at 10m and 10 at 50m"
- p. 15, Supplementary §C, paragraph "Performance on rain and night data", left column, lines 14-16: the dataset annotates "any scene with raindrops on the windshield as rainy", whether or not it is raining

### how it relates to our design

nuScenes is the dataset we train and evaluate on: its lidar gives our labels, its radar gives two input channels, and its night and rain scenes give the per-condition evaluation. The paper's synchronized cameras and lidar, and its release of intermediate radar frames, are what our lidar projection and 5-sweep radar residual rely on.

- [main.tex:127](../latex/main.tex:127): "The method needs lidar, camera sequences, calibrated radar, and condition labels in the same recording"; nuScenes is the one in use. [main.tex:129](../latex/main.tex:129) is the stub this sheet fills.
- p. 4, §2, paragraph "Sensor synchronization", left column, lines 1-4: camera exposure is "triggered when the top lidar sweeps" across the center of the camera's FOV, giving good lidar-camera alignment. p. 14, Figure 15 caption (left column): "We project the lidar points in an image plane". Our use: [main.tex:54](../latex/main.tex:54) (labels from lidar projected into the camera image) and [main.tex:174](../latex/main.tex:174) (lidar returns within 1-80 m; note Table 2 on p. 4 lists the lidar range as ≤ 70m).
- p. 4, §2, paragraph "Data annotation", right column, lines 9-13: intermediate sensor frames released, radar at 13Hz. Our use: [main.tex:166](../latex/main.tex:166) (radar residual over 5 sweeps) and [main.tex:244](../latex/main.tex:244) (150 radar points per image over 5 sweeps on mini, against 3,613 lidar points at [main.tex:243](../latex/main.tex:243)).
- p. 1, §1, ¶2, right column, lines 13-14: radar returns sparser and "less precise in terms of localization". Our observation: [main.tex:230](../latex/main.tex:230) (radar residual about five times the lidar-measured error; 13.5% of mini images have no radar fit).
- p. 4, §2, paragraph "Scene selection", right column, lines 10-13: expert annotators "write textual descriptions or captions for each scene", and the example given ends in the word rain. Related in our report: [main.tex:209](../latex/main.tex:209) (D11, a substring rule for scenes like "night, after rain") and [main.tex:319](../latex/main.tex:319) (compare the cache's keyword tags against the BEVCar lists). main.tex does not say which nuScenes field the tags are read from.

### notes for the writer

- The 700/150/150 split is not in the paper. The only statement about splits is p. 15, Supplementary §C, paragraph "Performance on rain and night data", left column, lines 3-5: "The dataset splits (train, val, test)" follow the same distribution over country, rain and night; no scene counts are given anywhere in pp. 1-16. Absence claim, not quotable. The counts in [main.tex:129](../latex/main.tex:129), [main.tex:261](../latex/main.tex:261) (631 + 69 training scenes) and [main.tex:121](../latex/main.tex:121) (150 val scenes) need another source (the nuScenes devkit or website), or should be stated as our own counts.
- The paper never says the radar lacks elevation; the words "elevation" and "height" are not used about radar anywhere in pp. 1-16. Absence claim, not quotable. What it does say: radar is sparser and less precise than lidar (p. 1), and the radars are mounted horizontally and "we calibrate the yaw angle using" static-object range rates, the only radar angle the paper mentions calibrating (p. 12, Supplementary §A, paragraph "Sensor calibration", bullet "Radar extrinsics", left column, lines 1-5). The "no elevation" statements at [main.tex:115](../latex/main.tex:115), [main.tex:129](../latex/main.tex:129) and [main.tex:230](../latex/main.tex:230) cannot be cited to this paper.
- The PDF is the arXiv version (p. 1 margin: "arXiv:1903.11027v5 [cs.LG] 5 May 2020") and prints no venue, so "CVPR 2020" in `references.bib` cannot be checked against it; title and the ten authors match. The paper's rain and night shares (19.4%, 11.6%) are shares of keyframes, not scenes, and "day/night/rain" is our grouping: the paper's own wording is times "(day and night)" and weather "(sun, rain and clouds)" (p. 4, left column, lines 3-4 below Table 2, the end of the "Drive planning" paragraph).

## post-review discussion with writer

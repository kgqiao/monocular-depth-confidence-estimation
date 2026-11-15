# literature review evidence: [Zenseact Open Dataset](../../Papers/2305.02008.pdf)

Alibeigi et al., ICCV 2023 · bib key `alibeigi2023zod` · report §2.4

## ai-generated summary review

### what the paper does

It releases ZOD, a driving dataset recorded over two years in 14 European countries with one front 8MP camera, three lidars and a high-precision GNSS/IMU, split into 100k single-image Frames, 1473 twenty-second Sequences and 29 Drives of a few minutes. It is released under CC BY-SA 4.0, which the authors call permissive because it allows commercial as well as research use.

- p. 1, Abstract, lines 4-7: "collected over two years in various European countries", covering 9× the area of existing datasets
- p. 2, §1, ¶3 (left column), lines 6-7: "100k independent Frames, 1473" twenty-second Sequences, and 29 Drives
- p. 4, Table 2, rows LiDARs and Camera: 1× Velodyne VLS128 (128 channels, "range up to 245m", 10Hz), 2× Velodyne VLP16 (16 channels, range up to 100m), 1× front RGB camera, HFOV 120°, 3848x2168
- p. 2, §1, ¶7 (right column), lines 1-4: "under the permissive CC BY-SA 4.0 license", research and commercial use "subject to the license terms"
- p. 7, §4.1, ¶1 (right column), lines 4-7: curated "ranging from the snowy parts of northern Sweden" to Italy, 14 countries in total

### the one result or idea that matters to us

The Sequences subset is the part with what a lidar-labeled, temporally aware evaluation needs: 20 seconds of 10Hz front-camera images with lidar scans from all three lidars for the whole clip, and per-datapoint calibration between sensors. Adverse weather is present but a minority: in Frames, rain is 15.7%, fog 2.1% and snow 2.0%, with about 19% at night.

- p. 6, §3.4, paragraph "Sequences" (left column), lines 1-2 and 7-10: 1473 scenes "each with a duration of 20 seconds"; blurred RGB images at 10Hz; lidar and GNSS/IMU "for the entire 20 seconds"
- p. 4, §3.1, paragraph "LiDAR data" (right column), lines 1-4: point clouds at 10Hz, each file holds "data from all three LiDAR" sensors
- p. 5, §3.2, ¶1 (right column), lines 9-13: per-datapoint calibrations with intrinsic and "extrinsic calibration) to transform data between any two" sensors
- p. 7, Figure 5(a) and (c): Frames weather clear 61.2%, cloudy 19.0%, rain 15.7%, fog 2.1%, snow 2.0%; time of day day 77.3%, night 19.0%, twilight 3.6%
- p. 3, Table 1, row ZOD Sequences: 6 European countries, 1473 annotated frames, 8.2 hours, 254k average lidar points, 120° camera

### how it relates to our design

Our method needs lidar, camera sequences, calibrated radar and condition labels in one recording; ZOD as described in this paper supplies the first two (in Sequences and Drives) and per-Frame weather metadata, but no radar. That is why the plan lists it as the fallback second dataset, to be used if the radar ablation shows the radar channel is not carrying the head.

- p. 4, §3.1, ¶1 (right column), lines 3-5: the full sensor list, "a high-resolution camera, front- and side-looking LiDARs," and GNSS/IMU sensors
- Absence claim, not quotable: ZOD has no radar in this paper. The word "radar" occurs twice in the whole PDF, neither about ZOD's sensors: p. 1, §1, ¶1, last line of the page, right column (range sensors in general) and p. 3, §2, ¶4 (right column), line 5 (nuScenes' "five radars"). Table 2 (p. 4) has no radar row.
- p. 6, §3.4, paragraph "Frames" (left column), lines 8-10 and 14-17: Frames carry ±1 second of lidar around one keyframe image, plus metadata including "weather conditions (e.g.," clear, rainy, foggy, snowy); p. 3, Table 1 caption, lines 2-3: "only the center image is provided"
- [literature-review.tex:61](../latex/sections/literature-review.tex:61): "The method needs lidar, camera sequences, calibrated radar, and condition labels in the same recording"
- [progress.tex:25](../latex/sections/progress.tex:25) and [progress.tex:34](../latex/sections/progress.tex:34): the temporal spread channel uses neighboring frames at ±1, ±2 (12 Hz); labels come from lidar returns within 1-80 m
- [plan.tex:22](../latex/sections/plan.tex:22): plan item "Second dataset": request STF / DENSE access; "if the radar ablation shows the radar channel is not carrying the head, evaluate on ZOD instead"

### notes for the writer

- Radar and the stub's logic. This paper cannot support "radar added in the 2025 release"; that needs its own source. Also, [literature-review.tex:71](../latex/sections/literature-review.tex:71) says ZOD is usable "if the radar ablation shows the channel matters", while [plan.tex:22](../latex/sections/plan.tex:22) says to use ZOD if the radar channel is *not* carrying the head. The two conditions point in opposite directions; pick one before filling the stub.
- "Nordic weather" is not the paper's wording and overstates it. The paper says 14 European countries "from the north of Sweden down to Italy" (p. 1, Figure 1 caption, last 2 lines), Sequences come from 6 countries (Table 1), and the weather shares in Figure 5 are for Frames only: snow 2.0%, fog 2.1%. The body gives no weather breakdown for Sequences, the subset with consecutive camera images. Frames have a single image each, so a temporal channel cannot be computed on them from this release (adding two seconds of camera frames to Frames is listed as future work, p. 9, §5, right column, "two seconds of supporting camera frames").
- License and camera. The license is CC BY-SA 4.0 (p. 2, §1, ¶7; reference [1]); "permissive" is the authors' word, and the SA part is share-alike, so say "CC BY-SA 4.0, which allows commercial use" rather than only "permissive". The camera is a single front camera with "wide-angle fish-eye lenses" (p. 5, paragraph "Camera data", left column, lines 1-2), unlike the six nuScenes cameras. Separately, the PDF in Docs/Papers is arXiv v2 (21 Oct 2023) and carries no ICCV mark, so the venue in `references.bib` cannot be checked against it; title and author order match.

## post-review discussion with writer

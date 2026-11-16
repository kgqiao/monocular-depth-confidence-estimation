# Deep Review Checklist

Triage of the Google Deep Review critique of the midterm report ([google-deep-review.md](google-deep-review.md), reviewed 2026-11-15). The review has 4 headline weaknesses, 16 numbered concerns, and about 15 minor corrections. They are sorted into two lists:

- **Checklist A, tonight:** wording, caption, table, and bibliography fixes whose facts are already in the code or the report. No run, no new code, no new commitment for the team.
- **Checklist B, team:** everything that needs a run, new code, a regenerated figure, or a team decision. For the poster session (November 30) and the final report (December 7).

Severity follows the review: **Critical** affects the validity of a claim or of the experimental setup, **Major** affects how a result is read, **Minor** is clarity or consistency. The § tag names the part of the review an item answers: §1 literature, §2 method, §3 results, §4 plan, with the item number in that part.

Progress, Checklist A: 29 / 29 applied on 2026-11-15 (branch `midterm/arteaga`). The edits add about 0.6 page of body text, so the built PDF goes from 10 to 11 pages; the body still ends on page 10 and page 11 holds references only.

## Checklist A: tonight (text only)

### Accuracy of claims

- [x] **A1 Critical.** "Training needs lidar and inference does not" is not true of the all-input head, because the temporal spread lifts each cell with the lidar fit. Qualify the sentence and name the lidar-free heads (`m1-features-only`, `m1-no-temporal`).
  - [introduction.tex:9](latex/sections/introduction.tex) · §2.1
- [x] **A2 Critical.** Say that `m1-features-only` needs no sensor fit at inference and comes within 0.0080 of all inputs, and that the heads reading the temporal spread are upper bounds for a deployed system.
  - [results.tex:52](latex/sections/results.tex) · §2.1
- [x] **A3 Major.** Say once that AbsRel is measured after the per-image lidar fit, so it is error of shape under the best scale and shift, not metric error.
  - [results.tex:13](latex/sections/results.tex) · §2.1
- [x] **A4 Major.** "The post-hoc methods that remain use no ground truth" is too strong. Add ConfidNet (Corbière et al., NeurIPS 2019) and SLURP (Yu et al., BMVC 2021), which train a second network on ground truth as we do, and soften the closing paragraph. Two new bib entries, checked against arXiv.
  - [literature-review.tex:22, 35](latex/sections/literature-review.tex), [references.bib](latex/references.bib) · §1.2
- [x] **A5 Major.** Figure 1 caption says the head reads "the four right-hand quantities". The temporal spread is middle-left, the top-right panel shows depth after the lidar fit (the head reads normalized disparity), and the bottom-left panel shows AbsRel before thresholding.
  - [main.tex:44](latex/main.tex) · §1.4
- [x] **A6 Major.** Plan item 11, "weight the depth by the confidence", reads as multiplying a depth by a probability, which shrinks depths instead of correcting them. Reword to a weight or a gate.
  - [plan.tex:38](latex/sections/plan.tex) · §4.1
- [x] **A7 Major.** The positive weight of 1.658 raises every logit by ln 1.658 = 0.506. Ranking metrics are unaffected; ECE is not. State that ECE is reported after subtracting it.
  - [progress.tex:45](latex/sections/progress.tex) · §2.2
- [x] **A8 Major.** The 0.001 tie band comes from one repeat pair. Call it a rough check, not a statistical test.
  - [results.tex:50](latex/sections/results.tex) · §3.3
- [x] **A9 Major.** 95.2% of labeled cells have no radar cell, so a tie in the loss over all labeled cells does not show whether the head uses radar where radar exists.
  - [results.tex:54](latex/sections/results.tex) · §3.2
- [x] **A10 Minor.** Say why the gradient method of Hornauer and Belagiannis is not among the baselines: a backward pass per image, and the cache stores activations without the graph.
  - [literature-review.tex:22](latex/sections/literature-review.tex) · §1.1
- [x] **A11 Minor.** Stage 3 losses are not comparable across τ. Say the τ runs are compared by AUSE and AURG on the AbsRel itself.
  - [plan.tex:18](latex/sections/plan.tex) · §4.3

### Definitions and consistency

- [x] **A12.** Table 1, channel 64: "Depth" to "Norm. disparity", noted as the depth channel of the later tables. Larger values are nearer, which "depth" hides. · [progress.tex:33](latex/sections/progress.tex) · §1.3
- [x] **A13.** Table 1, channel 65: "std divided by mean (σ/μ)", over the keyframe and its neighbors, up to 5 frames. Verified in `confhead/src/confhead/cache/temporal.py:47, 143`. · §1.3, §2 minor 1
- [x] **A14.** Table 1 caption: define d-hat (depth the radar fit gives the model's disparity at a radar point) and z (the point's range). Verified in `dataset.py:48-50`. · §1.3
- [x] **A15.** Labels paragraph: AbsRel is computed at each lidar return from the disparity sampled at its own pixel, and the fitted inverse depth is clipped to the 1 to 80 m range before inversion. Verified in `labels.py:31`, `geometry.py:133`. · §2.1, §2.3
- [x] **A16.** Head paragraph: stride 1 and padding 1, so the grid size is kept. Verified in `head.py:36`. · §2 minor 1
- [x] **A17.** Dev split: the 3,000 images are a fixed random subset, not the first 3,000. Verified in `train.py:288`. · §2.4
- [x] **A18.** Width × height everywhere: 900 × 1600 to 1600 × 900 in the pipeline figure, 296 × 528 to 528 × 296 in D3, and "12 × 12 pixels" is of the 1600 × 900 camera image. · [pipeline.tex:24](latex/figures/pipeline.tex), [progress.tex:22, 107](latex/sections/progress.tex) · §2 minor 2
- [x] **A19.** Table 2 caption: stage 0b is an analysis and trains no head. · §2 minor 3
- [x] **A20.** BEVCar counts: a sample is one keyframe with six camera images, so 6,019 samples are 36,114 images. · [progress.tex:90](latex/sections/progress.tex) · §2 minor 4
- [x] **A21.** Table 3: replace the shorthand "Form B" with "scope form" and define it in the caption. · [progress.tex:107-108](latex/sections/progress.tex) · §2 minor 4
- [x] **A22.** Table 4: the caption explains "×2" (the grid, not a doubling of images), † added to 37.6%, and the radar residual row is labeled "(radar fit)". · [results.tex:21-37](latex/sections/results.tex) · §2 minor 4, §3 minor
- [x] **A23.** Table 5: `0` to `0.0000`. · [results.tex:67](latex/sections/results.tex) · §3 minor
- [x] **A24.** Throughput: 1,007 images/s is the training step alone, without loading. · [results.tex:48](latex/sections/results.tex) · §3 minor
- [x] **A25.** D9: the nearest-point rule already removes a bleed-through return when a nearer return shares its cell; the share that survives on the grid is not yet measured. · [results.tex:41](latex/sections/results.tex) · §3.4
- [x] **A26.** Sentence fragment in plan item 14 ("The deviations table ..."). · [plan.tex:45](latex/sections/plan.tex) · §4 minor
- [x] **A27.** Bibliography venues: HyDRa is ICRA 2025 (arXiv comments field); LiCROcc is IEEE Robotics and Automation Letters vol. 10 no. 1, January 2025 (header of the published PDF on the authors' lab site; arXiv lists no venue). The LiCROcc citation year changes from 2024 to 2025. · [references.bib:32-46](latex/references.bib) · §1 minor
- [x] **A28.** Table 1, channel 66: where the footprints of two radar points overlap, the nearest point wins the cell. Verified in `geometry.py:86-116`. · §1.3
- [x] **A29.** Introduction: the output is an error probability on a 132 × 74 grid, and the confidence is one minus it. · [introduction.tex:9](latex/sections/introduction.tex) · §2.3

## Checklist B: team (before the poster and the final report)

Ordered within each group by how much the final results depend on the item.

### B1. Experimental setup

- [ ] **Critical. Lidar in the inputs.** The temporal spread (channel 65) is built with the keyframe lidar fit, and the planned stage 2 run puts the lidar fit into the radar residual too. Choose one: (a) rebuild the temporal channel without lidar (a per-camera median scale and shift from train, the radar fit, or a metric checkpoint), which needs the neighbor passes again; or (b) keep it and present the lidar-free heads (`m1-features-only`, `m1-no-temporal`) as the deployable models, with the 68-channel head and the lidar-fit radar run as upper bounds. · §2.1
- [ ] **Critical. Resolve the label masks before scoring, not after.** D8 (CAM_BACK trunk), D9 (bleed-through), and D13 (no-fit frames) change the labels, so scores computed first must be computed again. Labels are built at load time, so no cache rebuild. For D8, also drop the trunk points from the lidar fit, where their large inverse depth has high leverage. For D13, drop the no-fit frames and report their count; labeling every cell wrong mixes a failed fit with a depth error. · §4.2, §3.4
- [ ] **Major. Decouple the split seed from the training seed.** `train.py:279` and `:288` take the dev scenes and the dev subset from `train.seed`, so `m1-seed1` trained on other scenes as well as from another initialization. Commit a fixed dev scene list, then run the stage 5 seeds on it. `m1-seed1` can still be compared on val only. · §3.1
- [ ] **Major. Repair the cache, then retrain and score on it** (Nathan Aday, in progress as of 2026-11-16). The report now says the cache will be repaired and the ten heads retrained before scoring. Three things to settle:
  - **Build at `patch_multiple = 2`.** The existing cache is on the 132 × 74 grid, but `confhead/configs/nuscenes-trainval.toml:20` and the guides still default to 4 (264 × 148, about 1.1 TB). A rebuild with the repository default gives a different fingerprint and a second cache, not a repair. Use the cache's own `config.toml`, then fix the default and `confhead/README.md:61-63`.
  - **How much is missing.** The index holds 149,410 of 204,894 images (73%), which the report attributes to five incomplete shards of 16. The first look on the cluster says one shard and about 10%. One shard of 16 is 6%, so the two do not agree yet; confirm the count from the index before and after the repair.
  - **If the repair slips,** report per-camera metrics and a camera-balanced average, since CAM_FRONT is cached at 62.5% and CAM_FRONT_LEFT at 80%. · §2.4
- [ ] **Major. Refresh every number that depends on the cache after the repair.** Table 4 (including its seven dashes) and the 25.1% / 37.6% / 1.658 label figures; Table 5 and Figure 3 (all ten dev losses, the 0.0004 repeat gap, every Δ); radar coverage (3.4%, 4.8%, 17.0%, 32%); the BEVCar entry counts (19,459 / 4,237 / 2,635); the 73% and 149,410 statements in the abstract, Section 3.2, D10, and the risks; the Run 1 split sizes (110,929 train, 12,150 dev). · §2.4
- [ ] **Major. Reorder the schedule.** Stages 4 and 5 wait on the scores of stages 1 to 3, but the plan trains them all by November 17 and scores on November 20. Train stages 2 and 3 while the scoring script is finished, score stages 0 to 3, then launch stages 4 and 5. · §4.2

### B2. Evaluation protocol

- [ ] **Major. Fix the definition of AUSE and AURG and write it down.** Three choices: the error on the curve (AbsRel, RMSE, or the binary error rate), one curve per image averaged over images or all cells pooled, and whether a curve is divided by its error at zero removal. The review recommends per image, then averaged. · §1.3, §2.5
- [ ] **Major. Bootstrap over scenes, paired.** Resample the 150 val scenes (111 day, 24 rain, 15 night), keep the images of a scene together, B = 1,000, and take intervals on the per-scene difference between two models. Resampling images or cells treats correlated frames as independent and gives intervals that are far too narrow. This replaces the 0.001 tie band. · §2.5, §3.3
- [ ] **Major. ECE with the logit correction.** Subtract ln(w+) before the sigmoid, and report ECE before and after (optionally with temperature scaling fitted on dev). One line in the scoring script. · §2.2
- [ ] **Major. AUROC and ECE pooled over all labeled cells of a condition.** If a per-image mean is also reported, leave out single-class frames and frames with no labels, and log how many per condition. This settles the open AUROC edge case. · §2.5
- [ ] **Major. Compare `run-v1-radar` with `run-v0-no-radar` on the radar-valid cells alone.** No retraining. It answers whether the head uses radar where radar exists. · §3.2
- [ ] **Major. AUSE per epoch.** On mini, AUSE was best after epoch 1 and the loss after epoch 8. Only `best.pt` and `last.pt` are saved, both from epoch 8, so the 16-epoch run has to save a checkpoint per epoch for this to be tested. · §2.2
- [ ] **Minor. Stage 3:** choose τ by AUSE and AURG, or by AUROC at a fixed evaluation τ of 0.10. · §4.3

### B3. Baselines (cheap: no backbone pass, no training)

- [ ] **Major. Two rankings without parameters,** from channels the cache already holds: normalized disparity (channel 64, drop the farthest cells first) and the raw radar residual on radar-valid cells (channel 66). They test whether the head learns more than a rescaling of distance or of radar disagreement. · §1.1
- [ ] **Major. Score the trained single-input heads** (`m1-depth-only`, `m1-temporal-only`, `m1-no-features`) with the others. · §1.1
- [ ] **Minor. Hornauer and Belagiannis gradient baseline.** Either implement it inside the flip pass (backpropagate the flip difference to `output_conv1` and pool the gradient to the grid) or keep the exclusion now stated in the literature review. · §1.1

### B4. Method experiments (optional, most useful first)

- [ ] **Soft or continuous target.** A hard threshold gives no reason to order a 0.11 error against a 2.5 error, which may be why the loss and AUSE disagree. Try min(AbsRel / τ_max, 1) with soft BCE, a Smooth-L1 regression, or a ranking loss. No cache change. · §2.2
- [ ] **Radar channel.** Clip the residual to [0, 1] or [0, 2] before standardizing (as first planned in D6); fit with RANSAC or Huber and a positive scale instead of plain least squares; project each radar point as a vertical pillar to stand in for the missing elevation; note that a labeled cell more than 3 cells from a radar cell is outside the head's 7 × 7 receptive field. · §2.3, §3.2
- [ ] **Label placement diagnostics.** How often do the points of one cell disagree on AbsRel > 0.10, and how does the 37.6% positive rate move under the mean, median, or maximum of a cell instead of the nearest point? · §2.3
- [ ] **Downstream use of the confidence.** One of: confidence-weighted least squares for the radar scale-and-shift fit, a confidence-weighted average of depth over the neighbor frames, or dropping the least confident cells before lifting to 3D and reporting the outlier rate at fixed coverage. · §4.1

### B5. Final report writing

- [ ] **Numbered equations** for the scale-and-shift fit, AbsRel, the clipping of the fitted inverse depth, AUSE, and AURG. · §1.3, §2.1
- [ ] **Regenerate Figure 1** at full page width with readable panel titles and a consistent colorbar label, and Figure 3(c) with the label "0.5562–0.5566" (`figures-src/plot_run_losses.py`). · §1.4, §3 minor
- [ ] **Literature.** Add Sudhakar et al. 2022 (uncertainty from motion) and Singh et al. 2023 (radar-camera depth). Add the PDFs and evidence sheets for ConfidNet and SLURP, which went in tonight with bib entries only. · §1.2
- [ ] **Length.** The body is 9 pages against the recommended 5 to 6. For a 6 to 8 page final report: move the cluster and file-format detail to an appendix, merge Tables 2 and 5 into one table with the evaluation scores, move the deviations table, the milestone list, and the timeline out of the body, and remove the references to team meetings and the syllabus. · §4.4
- [ ] **Throughput.** Explain why the run loaded at about 70 images/s against the 28 images/s estimate, once measured. · §3 minor

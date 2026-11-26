# Internal Rubric for the Final Report

The course gives no rubric for the final report, so this is the team's own. The standard is a paper that a reviewer at a computer vision or machine learning venue (CVPR, ICCV, ECCV, NeurIPS, the venues in our references) would read as sound and complete, at the scale of a one-semester MS course project. That standard asks for rigor and honesty, not for a state-of-the-art result: a negative result that is measured well scores higher than a positive one that is not.

The rubric has two parts. The **gates** are pass or fail, and the report is not submitted until all eight pass. The **scorecard** gives 100 points over eight categories and shows where the remaining effort should go.

## Part 1: Gates

Every gate must pass, whatever the score. Each one names the place to check.

| # | Gate | Where to check |
|---|---|---|
| 1 | **Claims have evidence.** Every claim in the abstract, the introduction, and the conclusion points to a number, a table, or a figure in the body. | Read the three and mark each claim; find its number in Section 5. |
| 2 | **One evaluation set.** All methods of one table are scored on the same validation images and the same labeled cells, and the caption gives the counts. | Captions of Tables 3 to 6; Section 4.4, "Evaluation set". |
| 3 | **Intervals over scenes.** Every headline comparison has a 95% interval from a paired bootstrap over scenes, and the scene counts (111 day, 24 rain, 15 night) are next to the per-condition numbers. | Section 4.4, "Intervals"; Tables 3 and 4. |
| 4 | **Lidar is disclosed.** The report says which heads use lidar at inference, names the heads that do not, and says that AbsRel is measured after the per-image lidar fit. | Sections 3.1 and 3.5; the "Lidar at inference" column of Table 3; Section 6. |
| 5 | **Negative results are shown.** A result that goes against us is in the same table as the ones that go for us: radar, and any score without training that matches a head. | Tables 3 and 5; Section 6. |
| 6 | **Nothing is left over from the midterm.** No number of the partial cache, no `\todo`, `\tbd`, or placeholder figure, and no language of plans, meetings, the syllabus, or the proposal in the body. | `grep -n '\\todo\|\\tbd\|\\placeholderfig' sections/*.tex main.tex` returns nothing; search the PDF for "plan", "will", "midterm". |
| 7 | **It can be reproduced.** The report names the repository commit, the checkpoints, and the configuration files, and every figure and table has a script or a command. | Appendix E; `figures-src/`. |
| 8 | **Format.** At most 8 pages of body in the ACL template; the build has no undefined reference or citation; the contribution statement is one page and all five members have confirmed it. | The PDF and the build log; `contributions.tex`. |

## Part 2: Scorecard

100 points. The target is **at least 85 in total, and no category below 70% of its points**. A category below 70% is fixed before any other category is polished.

For each category, find the description that fits best and give a score in its range.

### 1. Problem, Motivation, and Contributions (10 points)

Abstract and Section 1.

- **Full (9 to 10).** The first paragraph makes a reader outside the project care. The gap is specific: what existing depth models and existing uncertainty methods do not give. The research questions are stated as questions. Three contributions each point to a section, and each is something a reader can check. The headline result is in the introduction with its number.
- **Partial (6 to 8).** The motivation is clear, but the contributions are general ("we evaluate our method") or the headline result has no number.
- **Insufficient (0 to 5).** The introduction describes the project and not the problem, or it claims more than Section 5 shows.

### 2. Related Work and Positioning (10 points)

Section 2.

- **Full (9 to 10).** It covers depth foundation models, uncertainty for depth, learned failure prediction, radar-camera fusion, and adverse-condition datasets. Each paper gets what it does and the one fact we use. The section says exactly how our head differs from ConfidNet, SLURP, and the methods of Poggi et al., and why the baselines we left out were left out. It fits its 0.8 page.
- **Partial (6 to 8).** The coverage is complete, but the section lists papers without saying where we stand, or it runs well over its budget.
- **Insufficient (0 to 5).** A close prior work is missing, or a paper is described wrongly.

### 3. Method: Clarity and Correctness (15 points)

Section 3.

- **Full (13 to 15).** A reader could build the labels and the head again from the text. The fit, AbsRel, the label, the loss, and the confidence are numbered equations that match the code. One pipeline figure shows the whole method. Each design choice has its reason in a clause (the grid, the threshold, the class weight, the nearest-point rule). Notation and the names of the inputs are the same in the text, the tables, and the figures.
- **Partial (9 to 12).** The method is correct and complete, but a reader has to open the code for a detail, or the notation changes between sections.
- **Insufficient (0 to 8).** An equation does not match the code, or a step of the pipeline is missing.

### 4. Experimental Design and Rigor (20 points)

Section 4. This and the next category carry the most weight, as they do with a reviewer.

- **Full (17 to 20).** The comparison is fair and is fixed before the results are read: the data and its split, the scene and label counts of each condition, the training settings, and the definition of each metric with its three choices (the error on the curve, per image or pooled, normalized or not). The baselines are strong for their cost: the oracle and a random order as bounds, flip disagreement, and the scores that need no training (normalized disparity, the raw temporal spread). Intervals come from a paired bootstrap over scenes. Repeat noise is measured with more than one pair of runs, or the report says that it is not.
- **Partial (12 to 16).** The protocol is sound, but one baseline that costs nothing is missing, or the intervals are over images, or a metric is not fully defined.
- **Insufficient (0 to 11).** Methods are scored on different images, or there is no interval at all, or the only comparison is with a random order.

### 5. Results and Analysis (20 points)

Section 5.

- **Full (17 to 20).** There is a main table of all methods, a per-condition table, the input ablation, the design ablations, a calibration result, and qualitative examples with at least one failure. Each paragraph opens with its finding and then gives the numbers. A difference is called real only when its interval supports it. The text explains why a result came out as it did (for radar: coverage and missing elevation) and does not only repeat the table. Each ablation changes one thing.
- **Partial (12 to 16).** The tables are complete, but the text walks through them without drawing a conclusion, or one of the six pieces is missing, or a claim rests on a gap inside the noise.
- **Insufficient (0 to 11).** The results rest on the dev loss, or on one run, or the main table has holes.

### 6. Limitations and Honesty (10 points)

Section 6, and the wording of every claim.

- **Full (9 to 10).** The limits are named before a reviewer finds them: lidar in the labels and in the temporal input, AbsRel as an error of shape, labels only where lidar returns, 15 night and 24 rain scenes, one dataset and one backbone, the coarse grid, and the runs that were not trained. Each claim is as strong as its evidence and no stronger. Future work follows from the limits.
- **Partial (6 to 8).** The main limits are there, but one that changes how a result is read is missing, or the abstract is stronger than the discussion.
- **Insufficient (0 to 5).** No limitations section, or a claim of deployment that the lidar dependence contradicts.

### 7. Presentation (10 points)

The whole report.

- **Full (9 to 10).** The body is at most 8 pages and reads in one sitting. Every figure is readable when printed, with labeled axes and units. Every caption can be read without the text. Tables use the same run names, the same number of decimals, and arrows for the direction of each metric. The writing is plain and in the present tense. There are no build warnings about references.
- **Partial (6 to 8).** The content is clear, but a figure is too small to read, or a table needs the text to be understood, or the sections differ in style.
- **Insufficient (0 to 5).** Over length, or figures that are screenshots, or sections that read as a work log.

### 8. Reproducibility and Artifacts (5 points)

Appendix E and the repository.

- **Full (5).** The commit, the checkpoints, the configuration files, and the command for each table and figure are given. The checkpoints can be downloaded by a reader, and the model card matches the report.
- **Partial (3 to 4).** The code and checkpoints are named, but a table cannot be made again from one command, or the checkpoint repository is still gated.
- **Insufficient (0 to 2).** A result in the report has no code in the repository.

### Score Sheet

| Category | Points | Floor (70%) | Reviewer 1 | Reviewer 2 |
|---|---|---|---|---|
| 1. Problem, motivation, contributions | 10 | 7 | | |
| 2. Related work and positioning | 10 | 7 | | |
| 3. Method: clarity and correctness | 15 | 10.5 | | |
| 4. Experimental design and rigor | 20 | 14 | | |
| 5. Results and analysis | 20 | 14 | | |
| 6. Limitations and honesty | 10 | 7 | | |
| 7. Presentation | 10 | 7 | | |
| 8. Reproducibility and artifacts | 5 | 3.5 | | |
| **Total** | **100** | **85 target** | | |

## Part 3: How We Use It

- **Who scores.** Each section is scored by two teammates who did not write it. A reviewer writes one line per category: the score and the one change that would raise it most. Line numbers of the PDF make the comments exact, which is why `main.tex` keeps the `review` option until the end.
- **Order of work.** Gates first, then any category under its floor, then the total.
- **Disagreement.** When two reviewers differ by more than 3 points on a category, they talk before the author changes anything.

Proposed dates, for the team to confirm. The poster session is Monday, November 30, and the report is due Monday, December 7.

| Date | Milestone |
|---|---|
| Wednesday, December 2 | Results freeze: every `\tbd` in Sections 4 and 5 is a number; gates 2 and 3 pass |
| Thursday, December 3 | Full draft: no `\todo` in the body |
| Friday, December 4 | Rubric review: two reviewers per section fill the score sheet |
| Sunday, December 6 | Text freeze: all gates pass; switch `review` to `preprint`; contribution statement confirmed |
| Monday, December 7 | Submit |

## Known Risks to the Gates, as of November 25

These were found while the skeleton was built. Each is marked with a `\todo` at its place in the text.

- **Gate 2.** The flip baseline was scored on 26,207 validation images and the first head on 35,927. The first number matches the partial cache, so the two were very likely scored on different images. Rescore the flip baseline on the complete cache.
- **Gate 3.** `confhead/examples/build_results_table.py` bootstraps over images. Frames of one scene are correlated, so those intervals are too narrow. It needs a paired bootstrap over scenes.
- **Gates 2 and 5.** The ECE is not yet corrected for the class weight, and flip disagreement is not a probability, so its ECE and Brier score are not calibration numbers and should be left out of the tables.
- **Gate 6.** The label masks D8, D9, and D13 are open. They change the labels, so every score computed before they are settled has to be computed again.
- **Gate 1.** Only one of the 18 heads has scores so far, and those are in a worklog entry on the `kgqiao` branch that is not merged. No per-image score file of a head is committed yet.

# Figure Sources

One script per figure of the final report, with the data it reads. A figure in `../latex/figures/` that has no script here cannot be redrawn when a number changes, so the rubric asks for one each (gate 7).

## Convention

- `plot_<figure-label>.py` writes `../latex/figures/<figure-label>.pdf`. Vector PDF for plots, PNG only for images.
- The script reads its data from a file in this folder (a CSV of per-image scores, a `history.json`), not from a path on the cluster. Commit the data file if it is small; if it is large, commit the command that makes it.
- Text in a figure is readable at the printed size: a figure that spans one column is drawn about 3.2 inches wide, a full-width one about 6.5 inches.
- The functions in `confhead/src/evals/plot.py` (`sparsification_figure`, `reliability_figure`, `qualitative_figure`) return a figure for the caller to save. Call them from the scripts here.

## Planned Figures

| Label | Section | What it shows | Data | State |
|---|---|---|---|---|
| `fig:teaser` | title | One adverse-condition frame: image, lidar AbsRel, head confidence | one cache entry and one checkpoint | to do; the cache-entry image holds the place |
| `fig:pipeline` | 3 | The pipeline from raw data to the loss | `../latex/figures/pipeline.tex` (TikZ) | done |
| `fig:sparsification` | 5.1 | Sparsification curves of the main methods against the oracle and a random order | per-image scores of each method | to do |
| `fig:reliability` | 5.5 | Reliability diagram of the best head, before and after the logit correction | pooled reliability bins | to do |
| `fig:qualitative` | 5.6 | Image, lidar AbsRel, head confidence, and flip disagreement for day, rain, and night, with one failure | cache entries and checkpoints | to do |
| `fig:losscurve` | Appendix B | Train and dev loss per epoch of the 18 runs | `history.json` of each run in `csci566-depthconf/confhead` | to do |

## From the Midterm Report

`Docs/Midterm-Report/figures-src/plot_run_losses.py` draws the loss curves from a folder of `history.json` files and can be reused for `fig:losscurve`. Its `histories/` folder holds the ten runs of the partial cache, and those were not copied here: download the histories of the 18 runs of the complete cache instead.

`../latex/figures/run-losses.pdf` is the midterm figure of the partial-cache runs. Nothing in the report includes it; delete it once `fig:losscurve` is redrawn.

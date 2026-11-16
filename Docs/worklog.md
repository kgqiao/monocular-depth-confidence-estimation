# Training Team Work Log

A running record of what the training team has done. Add new entries at the bottom, under a date heading.

## Status

| Item | State |
|---|---|
| nuScenes v1.0 dataset on CARC | Done |
| Confidence head input cache (trainval) | Done, checked |
| Cache tutorial notebook | In progress: not yet run on CARC |
| Project wiki in `Docs/Wiki` | Done |
| Cache backup on Hugging Face | Upload job running |
| Confidence head training, Run 1 (trainval) | Done, on Hugging Face |
| Run tracking (model card, Trackio) | Done locally; upload from a CARC job not yet tested |

Data locations on CARC:

- Dataset: `/scratch1/naday/nuscenes`
- Cache: `/scratch1/naday/confhead-cache/nuscenes-trainval`

The dataset is more than 800 GB, so it cannot go into the PI's project folder. `/scratch1` also gives the fastest read access on CARC.

## 2026-10-19

### Downloaded the full nuScenes dataset

- Downloaded nuScenes v1.0 (trainval and test) to `/scratch1/naday`. The download took 7 hours.
- Script: `utilities/dataset/download_nuscenes.py`, run as a job with `download_nuscenes.sbatch`. It is adapted from [li-xl/nuscenes-download](https://github.com/li-xl/nuscenes-download).
- Changes from the original script:
  - The nuScenes login comes from an env file, so no credentials go into git.
  - A stopped download resumes, each archive must match its MD5, and extracted archives are skipped on a second run.
- Checked that the downloaded archives are the parts of nuScenes the project needs. nuScenes offers several datasets, and only some of them apply.

### Built the confidence head input cache

- Built the trainval cache on CARC with the `confhead` API. Config: `confhead/configs/nuscenes-trainval.toml`. Guide: `confhead/guides/cache-building-on-carc.md`.
- The build is complete. We checked the cache report a second time.

## 2026-10-20

### Started a cache tutorial notebook

- `confhead/notebooks/cache-tutorial.ipynb` shows how to look at the cache and documents the cache reading API.
- `confhead/guides/notebook-on-carc.md` tells how to open it on CARC.
- To do: run the notebook on CARC against the full cache.

### Added project wiki pages

- Copied six project pages into `Docs/Wiki`, plus `Training.canvas` (an Obsidian canvas).

### Wrote the Hugging Face backup

- `confhead cache upload` splits the cache into shards, compresses each shard with zstd, and uploads it to a Hugging Face dataset repository. `confhead cache download` restores it.
- Purpose: a backup of the cache, and team access if access to `/scratch1/naday` is too limited.
- Config: `confhead/configs/huggingface.toml`. Guide: `confhead/guides/cache-backup-on-huggingface.md`.

### Started the Hugging Face upload on CARC

- Started the shard, compress, and upload job (`confhead/examples/carc-upload.sbatch`) at 2:44 PM.
- To do: when the job ends, confirm that all shards are on Hugging Face and that a download restores a complete cache.

## 2026-10-27

- Setup subdirectory for evals at root level
- Wrote initial rejection-auc, and ece metric functions.
- Wrote main function for initiating metric tests, and template for filling in further evals code
- Set up synthetic tests for metrics, starting with sparsification curves
- To Do: Finish functions, start data loading logic, bootstrapping, connection to access cache data

## 2026-10-28

### Estimated the cost of training on trainval

- Ran the estimate as job 13527183 (log `confhead/logs/confhead-train-13527183.out`):

  ```
  sbatch --time=01:00:00 examples/carc-train.sbatch configs/train-trainval.toml --estimate
  ```

- Data: 631 train scenes (110,929 images, all used) and 69 dev scenes (3,000 of 12,150 images used). The train split has 76 night and 125 rain scenes.
- Labels: 25.1% of cells carry a lidar label. Of those, 37.6% have an AbsRel above `tau` = 0.10, which is inside the 5% to 70% gate. The class weight is 1.658.
- Speed: the cache loads 28 images/s and a training step takes 1,007 images/s, so loading sets the pace. One epoch takes about 66 minutes, and 8 epochs about 8.8 hours. That fits in the 12-hour limit of `carc-train.sbatch`.

### Started the first training job

- Started Run 1 on the full trainval cache, writing to `/scratch1/naday/confhead-runs/run1`:

  ```
  sbatch examples/carc-train.sbatch configs/train-trainval.toml
  ```

- To do: when the job ends, check the dev loss of each epoch and that the run folder holds the trained head.

## 2026-11-06

### Finished Run 1

- Run 1 trained all 8 epochs in 3.5 hours, against the estimate of 8.8 hours. The dev loss fell in every epoch, so the best head is epoch 8, with a dev loss of 0.5562.
- Uploaded the run folder to the model repository [`nadayusc/confhead`](https://huggingface.co/nadayusc/confhead), under `run1/`.

### Tied the upload and the run tracking into the training job

- `carc-train.sbatch` now passes `--upload`: the job uploads the run folder after each epoch and once more at the end. The model card of the repository is built from the `config.json` and `history.json` of every run there, so the same runs always give the same card.
- Each run folder also holds `steps.json`, the mean train loss five times per epoch. A new setting, `output.group`, names the family of a run, such as `baseline` or `radar`.
- `examples/sync_trackio.py` rebuilds a local Trackio project from the repository. Guide: `confhead/guides/tracking-runs.md`.
- To do: test the upload from a CARC job by submitting Run 1 again. The job finds 8 of 8 epochs, trains nothing, and uploads.
- To do: decide how the team shares the dashboard: each person runs the sync, or a hosted Trackio Space.

## 2026-11-14

### Revised and fixed various small on evals, testing

- Move evals/ to confhead/src/evals/. `uv run` puts confhead/src on the import path, so `from evals.metrics import ...` now resolves, and tests/test_metrics.py no longer stops pytest at collection.
- metrics.py: np.trapz -> np.trapezoid (np.trapz is gone in the project's NumPy 2.5.4)
- run_evals.py: remove a stray line continuation that was a SyntaxError
- Checked with `uv run pytest -q`: no collection error; 66 passed, 5 skipped. The 4 failures are in test_backup.py and happen on Windows only.

### Finished the sparsification curves and their tests

- `compute_sparsification_curves` now returns four values: fractions, method curve, oracle curve, and the random curve (the mean error at every fraction), which `calculate_ause_and_aurg` needs.
- `tests/test_metrics.py`: a perfect score gives AUSE 0, a random score gives AURG near 0, a reversed score gives a negative AURG, and a constant error gives flat curves. 4 passed.

### Started the rejection AUROC and ECE fixes

- Added scikit-learn (1.9.1) with `uv add scikit-learn`; `calculate_rejection_auc` imports it as `sklearn`.
- To do: return NaN from `calculate_rejection_auc` when every pixel is correct or every pixel is wrong (`roc_auc_score` raises then, and per-image scoring will hit such images); count a confidence of exactly 1.0 in the last ECE bin; tests in `tests/test_calibration.py`.

### Bev 
- Bevcar work today - Checked with `uv run pytest -q`: no collection error; 66 passed, 5 skipped. The 4 failures are in test_backup.py and happen on Windows only.
- Created confhead/src/evals/bevcar_splits.py — new file with the three scene name lists (DAY_SCENES, RAIN_SCENES, NIGHT_SCENES) and the BEVCAR_CONDITION lookup dict
- Modified confhead/src/evals/run_evals.py — added defaultdict accumulator, wired in BEVCAR_CONDITION.get(scene_name) inside the data loop, and added the per-condition metrics loop after it
- Created confhead/examples/compare_condition_tags.py — audit script to compare cache keyword tags vs BEVCar lists on CARC
- Created confhead/examples/label_count_table.py — script to count scenes, keyframes, labeled cells, and AbsRel > tau per BEVCar group

### Evals team set future checklist items

Varad:
- Download all three trained models as of today from Nathan A and Nathan B
- Build flip comparison py file, start on CARC (mirror image, run depth model, flip result, use where two depths diagree as a no-training confidence score)

Katherine:
- Finish the AUROC, ECE tests
- Write scoring code to load trained model #1 to run on test images, score each image with metrics, save results 
- Run scoring script on all test images on CARC, after 50 image test run succeeds

Nathan A/B:
- Have all models trained

DUE BY TUE 17th


----- Run scoring on all models + condition splits, write analyses
Katherine:
- Run scoring script on all test images for all 3 trained models downloaded (and any new models)
- Add all results to run_matrix.md
- Write analysis and make charts based on three models' scores (sparsification curves, comparing models with oracle and random)

Varad:
- Count how many pixels with a LIDAR answer each of day/rain/night has, utilizes scene list, goes into table
- Build day/rain/night results table from scores from all models, scene lists, flip scores. Show which differences are real

Katherine/Varad:
- Look at eval results and decide top promising models

DUE BY Fri Nov 20


---- Decide from most promising models and maybe run additional models

Nathans:
- Run additional models if needed

DUE BY Sat Nov 21


--- Decide on the project before/after results

Katherine:
- Create frame selector to rank whole images by confidence. Check whether dropping the whole images lowers the error of the rest. 

Varad:
- Show use of confidence (i.e. weigh depth by it) with one number that improves (maybe team decides which taht is?)
- NuScenes drive before and after (depth estimation with and without)

Due by 24th




---- Writing final report + prepare for project clinic Nov 30

Katherine:
- Add to final report: charts, main results table of every model

Varad:
- Add to final report: day/rain/night results, comparisons with simple scores, lidar data counts, limits\

Alejandro/everyone:
- Finalize and wrap up report
- Prepare for project clinic on Nov 30

--- Final Report due Dec 7


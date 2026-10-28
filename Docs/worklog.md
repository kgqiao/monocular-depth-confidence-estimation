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
| Confidence head training, Run 1 (trainval) | Training job running |

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

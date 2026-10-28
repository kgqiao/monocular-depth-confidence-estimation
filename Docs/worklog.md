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
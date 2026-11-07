# Training Runs and Their Configs

This guide lists the confidence-head training runs and says what each setting of a run means. It is written for the eval team, who score the heads. The steps to train a run are in [training-on-carc.md](training-on-carc.md).

## What a Run Is

A run trains one confidence head on the v1.0-trainval cache, on top of a frozen Depth Anything V2 ViT-B. The head outputs one logit per grid cell, `p_error`: the probability that the model's depth in that cell is wrong by more than `tau` against lidar.

| Run | Radar channels | Input channels | Where | State |
|---|---|---|---|---|
| `run-v0-no-radar` | off | 66 | `nathancbao/confhead` (private) and `/scratch1/ncbao/confhead-runs/` | Done: 8 of 8 epochs in 2 h 44 min, best dev loss 0.5562 (epoch 8) |
| `run1` | on | 68 | [`nadayusc/confhead`](https://huggingface.co/nadayusc/confhead/tree/main/run1) | Done: 8 of 8 epochs, best dev loss 0.5562 (epoch 8) |
| `run-v1-radar` | on | 68 | `nathancbao/confhead` (private) and `/scratch1/ncbao/confhead-runs/` | Done: 8 of 8 epochs in 5 h 29 min, best dev loss 0.5566 (epoch 8). A repeat of `run1`, with the same settings |

`run-v0-no-radar` and `run1` differ in one setting, `inputs.radar`. Everything else is the same, including the seed (0), the dev scenes, and the labels, so a difference in score between them comes from radar alone.

`run1` and `run-v1-radar` have the same settings and differ by 0.0004 in best dev loss. That is the noise from repeating a run. Radar on (0.5562, 0.5566) and radar off (0.5562) are inside it. Dev loss does not say how good the confidence is, so the radar question is for AUSE and AURG.

The commands for the two runs of this account:

```sh
sbatch examples/carc-train.sbatch configs/train-trainval.toml --set inputs.radar=false --set output.name=run-v0-no-radar
sbatch examples/carc-train.sbatch configs/train-trainval.toml --set output.name=run-v1-radar
```

## Files in a Run Folder

| File | What it holds |
|---|---|
| `best.pt` | the head of the epoch with the lowest dev loss. Score this one. |
| `last.pt` | the head of the last epoch, with the optimizer and the schedule, to continue a run |
| `config.json` | every setting of the run, the dev scenes, the class weight, the cache fingerprint, the git commit |
| `history.json` | the train and dev loss of each epoch |

## The Settings

[`configs/train-trainval.toml`](../configs/train-trainval.toml) holds every setting. A run saves its own copy in `config.json`.

| Setting | Value | What it is |
|---|---|---|
| `data.cache` | `/scratch1/$USER/confhead-cache/nuscenes-trainval` | the cache that is read (149,410 entries, fingerprint `99d44ff8c71c5355`) |
| `inputs.features`, `inputs.depth`, `inputs.temporal` | true | the 64 decoder features, the normalized disparity, and the temporal spread |
| `inputs.radar` | false in v0, true otherwise | the two radar channels: the residual and the validity mask |
| `inputs.labels.align` | `lidar` | the label comes from the lidar fit of the model's disparity |
| `inputs.labels.tau` | 0.1 | a cell counts as wrong when AbsRel is above 0.1 |
| `inputs.labels.min_depth`, `max_depth` | 1.0, 80.0 m | the depth range of the labels |
| `inputs.radar_cfg.align` | `radar` | the radar residual uses the radar fit, the fit that a car could make without lidar |
| `inputs.radar_cfg.radius` | 1 | each radar point also fills the cells within 1 cell |
| `inputs.radar_cfg.max_sweeps` | 0 | use every cached radar sweep |
| `head.widths` | [64, 64, 32] | three 3x3 conv layers, then a 1x1 conv to one logit per cell |
| `train.epochs` | 8 | |
| `train.batch_size` | 8 | |
| `train.lr` | 0.001 | |
| `train.weight_decay` | 0.0001 | |
| `train.samples_per_epoch` | 0 | 0 means every train image in each epoch |
| `train.dev_fraction` | 0.1 | the share of train scenes held out as the dev set |
| `train.dev_limit` | 3000 | at most 3,000 dev images are used |
| `train.stats_sample` | 256 | the images used for the per-channel input statistics |
| `train.seed` | 0 | |

From the log of the run: 631 train scenes (110,929 images), 69 dev scenes (12,150 images, of which 3,000 are used; 8 night and 13 rain scenes), 25.1% of the cells labeled, a positive rate of 37.6%, and a class weight of 1.6575. The dev scenes are the same in every run with these settings.

## What Eval Keeps the Same

- Score on the val split of the cache, the official nuScenes val with 150 scenes. The dev set is part of the train split and is not the val split.
- Use the labels of the training: `tau` 0.1 and lidar alignment. `load_head("<run>/best.pt")` returns the head, the `HeadInputs` it was trained with, and the saved fields. Build the inputs from that `HeadInputs`, so that the channels match.
- [`notebooks/eval-template.ipynb`](../notebooks/eval-template.ipynb) loads `best.pt` and predicts `p_error` for val images.

## Get the Files

- From Hugging Face: `uv run hf download nadayusc/confhead <run>/best.pt --local-dir .` for `run1`, and the same with `nathancbao/confhead` for `run-v0-no-radar` and `run-v1-radar`. The `nathancbao/confhead` repository is private, so ask Nathan B to add your Hugging Face username first.
- From CARC: the folder of a run. A scratch folder is private to its owner, so it has to be copied or shared first.

## Before You Quote a Result

- The dev loss is a weighted cross-entropy on the dev images. It does not say how well the confidence works: AUSE and AURG do.
- The cache was built with the relative Depth Anything V2 ViT-B checkpoint, and the labels use a per-image lidar fit. The Run 1 plan named the metric checkpoint. This is open question 2 of [CS566 Implementation Deviations and Open Questions](../../Docs/Wiki/CS566%20Implementation%20Deviations%20and%20Open%20Questions.md).
- The dev set has 8 night and 13 rain scenes, so dev numbers for night and rain are noisy.
- The cache holds 149,410 entries, and the nuScenes keyframes of the 850 scenes make 204,894 camera images. The reason for the difference is not confirmed.

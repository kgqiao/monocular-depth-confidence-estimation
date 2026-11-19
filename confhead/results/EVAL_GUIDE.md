# Guide for Eval: which run is which

This repository holds confidence heads trained on top of a frozen Depth Anything V2 (ViT-B) on nuScenes. A head predicts `p_error` for every grid cell: the probability that the depth of that cell is wrong by more than 10% against lidar. Each folder is one training run, trained on the complete cache. Runs on the earlier partial cache are under `--revision cache-149k`, and their dev losses are not comparable with these. Every run changes one thing from the first run, so a difference between two runs comes from that one change.

The model card (`README.md`) lists every run with its settings and its loss per epoch. This file says what to score and how.

## The Runs

All runs have the same settings except the one in the second column: seed 0, 8 epochs, batch size 8, learning rate 0.001, head widths [64, 64, 32], labels from the lidar fit with `tau` 0.1. The same cache and the same train and dev scenes are used, except in `m1-seed1`. The three runs marked "other labels" cannot be compared with the rest on dev loss.

| Run | What is different | Input channels | Best dev loss |
|---|---|---:|---:|
| `run-v1-radar` | nothing: all inputs (decoder features, depth, temporal spread, radar) | 68 | 0.5589 |
| `run-v1-radar-repeat` | nothing: the same settings again, to measure the repeat noise | 68 | 0.5581 |
| `run-v0-no-radar` | no radar channels | 66 | 0.5580 |
| `m1-no-depth` | no depth channel | 67 | 0.5607 |
| `m1-no-temporal` | no temporal spread | 67 | 0.5648 |
| `m1-features-only` | only the decoder features (no depth, temporal, or radar) | 64 | 0.5683 |
| `m1-no-features` | no decoder features: depth, temporal spread, and radar only | 4 | 0.6129 |
| `m1-depth-only` | the depth channel alone | 1 | 0.6368 |
| `m1-temporal-only` | the temporal spread alone | 1 | 0.6713 |
| `m1-seed1` | all inputs, random seed 1 (this also picks other dev scenes) | 68 | 0.5622 |
| `m2-radar-lidar-fit` | the radar residual uses the lidar fit, not the radar fit | 68 | 0.5534 |
| `m2-radius0` | each radar point fills only its own cell (radius 0) | 68 | 0.5591 |
| `m2-radius2` | each radar point fills the cells within 2 (radius 2) | 68 | 0.5581 |
| `m2-sweeps1` | the keyframe radar sweep only | 68 | 0.5582 |
| `m6-quarter-data` | a quarter of the train images per epoch (38,000) | 68 | 0.5664 |
| `m3-tau005` | `tau` 0.05 (other labels) | 68 | 0.4362 |
| `m3-tau02` | `tau` 0.2 (other labels) | 68 | 0.6349 |
| `m6-maxdepth50` | label depth range up to 50 m (other labels) | 68 | 0.5633 |

`m1-` to `m6-` are the stages of the run matrix: [`run-matrix.md`](https://github.com/kgqiao/monocular-depth-confidence-estimation/blob/main/confhead/guides/run-matrix.md) in the project repository lists the planned runs and their exact settings.

## What to Score First

1. `run-v1-radar` and `run-v0-no-radar`: does radar help?
2. `m1-no-features`, `m1-no-temporal`, `m1-no-depth`, `m1-features-only`: what does each input add?
3. `m1-depth-only` and `m1-temporal-only`: simple cues next to the learned head.
4. `m2-radar-lidar-fit` first of the new runs (lowest dev loss of all, 0.0055 under the baseline), then `m2-radius0`, `m2-radius2`, `m2-sweeps1`: radar settings. Score on the labeled cells that have radar too, because radar covers few cells.
5. `m6-quarter-data`: how much the head loses with a quarter of the data.
6. `m3-tau005`, `m3-tau02`, `m6-maxdepth50`: score against the labels of their own `tau` or depth range, from their `config.json`, never against the `tau` 0.1 labels.
7. `m1-seed1`: score it like the others. Its use is the noise between seeds. Do not compare its dev loss with the others, because its dev scenes are different.

## Access

This repository is gated. Log in (`hf auth login`) with an account that an admin added, then download.

## How to Load a Run

```sh
hf download csci566-depthconf/confhead <run>/best.pt --local-dir .
```

```python
from confhead import load_head
head, inputs, fields = load_head("<run>/best.pt")
```

`best.pt` is the head of the epoch with the lowest dev loss (epoch 8 in most runs, epoch 7 in `run-v1-radar` and `m6-quarter-data`). `config.json` holds every setting of the run, and `history.json` the loss of each epoch.

The runs have different input channels, so build the input of each run from the `inputs` that `load_head` returns, never from a fixed list of channels. The notebook [`eval-template.ipynb`](https://github.com/kgqiao/monocular-depth-confidence-estimation/blob/main/confhead/notebooks/eval-template.ipynb) shows the steps.

## What to Keep the Same

- Score on the official val split of the cache (150 scenes), not on the dev set. The dev set is part of the train split.
- Use the training labels of each run: `tau` 0.1 and the lidar alignment, except in `m3-tau005`, `m3-tau02` and `m6-maxdepth50`, which use their own (see their `config.json`).
- Score every run the same way, so that the runs are comparable.

## Before You Quote a Result

- The "best dev loss" above is a weighted cross-entropy on 3,000 dev images from the dev scenes of the train split (listed in each `config.json`). It tells whether a run learned. It does not tell how good the confidence is: AUSE and AURG do.
- Two runs with the same settings (`run-v1-radar` and `run-v1-radar-repeat`) differ by 0.0008 in dev loss. A difference below about 0.001 between two runs is not a real difference.
- Radar covers few cells (measured on the earlier partial cache): at the training setting it marks 3.4% of the grid cells, and only 4.8% of the lidar-labeled cells have a radar cell on them. The radar fit fails on 17% of the images, and those images get zero radar input.
- The cache was built with the relative Depth Anything V2 ViT-B checkpoint, and the labels use a per-image lidar fit.

## License

The heads were trained on [nuScenes](https://www.nuscenes.org/), which is under CC BY-NC-SA 4.0 and the nuScenes terms of use. The same terms apply here: non-commercial use only.

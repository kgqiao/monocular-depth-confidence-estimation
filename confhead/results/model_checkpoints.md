# Model Checkpoints

This page lists every training run of the confidence head, with a link to its checkpoint, for anyone picking a run to evaluate or compare. To see what each run tests and what the losses mean, read [the run matrix](../guides/run-matrix.md).

> [!IMPORTANT]
> Every run below trained on the complete cache (204,894 entries) on 2026-11-17. Earlier runs with the same names trained on a partial cache. Their files are in the repository under the tag `cache-149k`, and their losses cannot be compared with the ones here. A score from before 2026-11-17 belongs to those earlier files.

## Resource Links

| Resource | What it holds |
|---|---|
| [`guides/run-matrix.md`](../guides/run-matrix.md) | The run matrix: every run, what it changes from [`configs/train-trainval.toml`](../configs/train-trainval.toml), its state, and the stage 1 findings |
| [`configs/run-matrix.toml`](../configs/run-matrix.toml) | The manifest that `confhead train` reads: the name and settings of each run |
| [`csci566-depthconf/confhead`](https://huggingface.co/csci566-depthconf/confhead) | The checkpoints of every run of the matrix, and the upload target of the training jobs. Its model card lists the settings and per-epoch losses of each run. The repository is gated: log in, and ask for access (or be added by an admin) |
| [`csci566-depthconf/confhead` `EVAL_GUIDE.md`](https://huggingface.co/csci566-depthconf/confhead/blob/main/EVAL_GUIDE.md) | For the Eval team: which run is which, what to score first, and how to load a run. [`EVAL_GUIDE.md`](EVAL_GUIDE.md) in this folder is a copy. Both still give the dev losses of the partial-cache runs |
| [`nathancbao/confhead`](https://huggingface.co/nathancbao/confhead) | The earlier home of the stage 0 and 1 runs on the partial cache (public) |
| [`nadayusc/confhead`](https://huggingface.co/nadayusc/confhead) | `run1`, the first trainval run, on the partial cache |
| [`nathancbao/confhead-charts`](https://huggingface.co/spaces/nathancbao/confhead-charts) | Trackio loss curves of the stage 0 and 1 runs on the partial cache, a snapshot from 2026-11-15 without `run1` |

## Run Catalogue

Every run trained on the complete `nuscenes-trainval` cache (fingerprint `99d44ff8c71c5355`, 152,124 train images) for 8 epochs, and each folder holds `best.pt`, `last.pt`, `config.json`, `history.json`, and `steps.json`. The dev loss only shows that a run learned. AUSE and AURG (the areas under the sparsification-error and random-gain curves), which the Eval team scores on the val split, decide which run is best. Two runs with the same settings differ by 0.0008, so treat a gap below about 0.001 as a tie.

| Run | What changes from the baseline | Inputs (channels) | Best dev loss | Checkpoint |
|---|---|---:|---:|---|
| `run-v1-radar` | nothing: the baseline | 68 | 0.5589 | [`csci566-depthconf/confhead/run-v1-radar`](https://huggingface.co/csci566-depthconf/confhead/tree/main/run-v1-radar) |
| `run-v1-radar-repeat` | nothing: a repeat of the baseline | 68 | 0.5581 | [`csci566-depthconf/confhead/run-v1-radar-repeat`](https://huggingface.co/csci566-depthconf/confhead/tree/main/run-v1-radar-repeat) |
| `run-v0-no-radar` | no radar | 66 | 0.5580 | [`csci566-depthconf/confhead/run-v0-no-radar`](https://huggingface.co/csci566-depthconf/confhead/tree/main/run-v0-no-radar) |
| `m1-no-depth` | no depth channel | 67 | 0.5607 | [`csci566-depthconf/confhead/m1-no-depth`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-no-depth) |
| `m1-no-temporal` | no temporal spread | 67 | 0.5648 | [`csci566-depthconf/confhead/m1-no-temporal`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-no-temporal) |
| `m1-features-only` | decoder features alone | 64 | 0.5683 | [`csci566-depthconf/confhead/m1-features-only`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-features-only) |
| `m1-no-features` | no decoder features | 4 | 0.6129 | [`csci566-depthconf/confhead/m1-no-features`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-no-features) |
| `m1-depth-only` | depth channel alone | 1 | 0.6368 | [`csci566-depthconf/confhead/m1-depth-only`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-depth-only) |
| `m1-temporal-only` | temporal spread alone | 1 | 0.6713 | [`csci566-depthconf/confhead/m1-temporal-only`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-temporal-only) |
| `m1-seed1` | seed 1, which also picks other dev scenes | 68 | 0.5622 | [`csci566-depthconf/confhead/m1-seed1`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-seed1) |
| `m2-radar-lidar-fit` | radar residual from the lidar fit | 68 | 0.5534 | [`csci566-depthconf/confhead/m2-radar-lidar-fit`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m2-radar-lidar-fit) |
| `m2-radius0` | radar radius 0 | 68 | 0.5591 | [`csci566-depthconf/confhead/m2-radius0`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m2-radius0) |
| `m2-radius2` | radar radius 2 | 68 | 0.5581 | [`csci566-depthconf/confhead/m2-radius2`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m2-radius2) |
| `m2-sweeps1` | the keyframe radar sweep only | 68 | 0.5582 | [`csci566-depthconf/confhead/m2-sweeps1`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m2-sweeps1) |
| `m3-tau005` | `tau` 0.05, so other labels | 68 | 0.4362 | [`csci566-depthconf/confhead/m3-tau005`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m3-tau005) |
| `m3-tau02` | `tau` 0.2, so other labels | 68 | 0.6349 | [`csci566-depthconf/confhead/m3-tau02`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m3-tau02) |
| `m6-maxdepth50` | label depth range up to 50 m (other labels) | 68 | 0.5633 | [`csci566-depthconf/confhead/m6-maxdepth50`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m6-maxdepth50) |
| `m6-quarter-data` | 38,000 train images per epoch, a quarter | 68 | 0.5664 | [`csci566-depthconf/confhead/m6-quarter-data`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m6-quarter-data) |

`run-v1-radar` and `run-v1-radar-repeat` share every setting, the dev scenes, and the code. They are the one repeat pair on the complete cache, and the 0.0008 noise figure comes from them. `run1`, the first trainval run, trained on the partial cache and is in [`nadayusc/confhead`](https://huggingface.co/nadayusc/confhead/tree/main/run1).

Do not compare the dev loss of a `tau` run or of `m6-maxdepth50` with the others: they use other labels. Stages 4 and 5 of the run matrix (learning rate and head width, final seeds) have not run yet.

To load a partial-cache checkpoint, add `--revision cache-149k` to the download command below.

## Loading a Checkpoint

```sh
hf download csci566-depthconf/confhead m1-no-temporal/best.pt --local-dir checkpoints
```

```python
from confhead import load_head

head, inputs, info = load_head("checkpoints/m1-no-temporal/best.pt")
```

The runs take different input channels, so build each run's input from the `inputs` that `load_head` returns. [`notebooks/eval-template.ipynb`](../notebooks/eval-template.ipynb) shows the steps.

## Adding a Run

1. Add a `[[run]]` to [`configs/run-matrix.toml`](../configs/run-matrix.toml) and train it with `confhead train run`, as [training-on-carc.md](../guides/training-on-carc.md) describes. The job uploads the run to `csci566-depthconf/confhead`.
2. Add a row to the table above, and fill in the state and dev loss in the run matrix.

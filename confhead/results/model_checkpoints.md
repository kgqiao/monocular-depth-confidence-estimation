# Model Checkpoints

This page lists every training run of the confidence head, with a link to its checkpoint, for anyone picking a run to evaluate or compare. To see what each run tests and what the losses mean, read [the run matrix](../guides/run-matrix.md).

> [!IMPORTANT]
> Every run below trained on the partial cache: 149,410 of the 204,894 entries. All of them are being trained again on the complete cache with `confhead train` and will replace these under the same names. Until this page is updated, check the model card of the repository for the state of a run: a run on the complete cache has more than 150,000 train images.

## Resource Links

| Resource | What it holds |
|---|---|
| [`guides/run-matrix.md`](../guides/run-matrix.md) | The run matrix: every run, what it changes from [`configs/train-trainval.toml`](../configs/train-trainval.toml), its state, and the stage 1 findings |
| [`configs/run-matrix.toml`](../configs/run-matrix.toml) | The manifest that `confhead train` reads: the name and settings of each run |
| [`csci566-depthconf/confhead`](https://huggingface.co/csci566-depthconf/confhead) | The checkpoints of every run of the matrix, and the upload target of the training jobs. Its model card lists the settings and per-epoch losses of each run. The repository is gated: log in, and ask for access (or be added by an admin) |
| [`csci566-depthconf/confhead` `EVAL_GUIDE.md`](https://huggingface.co/csci566-depthconf/confhead/blob/main/EVAL_GUIDE.md) | For the Eval team: which run is which, what to score first, and how to load a run. [`EVAL_GUIDE.md`](EVAL_GUIDE.md) in this folder is a copy |
| [`nathancbao/confhead`](https://huggingface.co/nathancbao/confhead) | The earlier home of the stage 0 and 1 runs on the partial cache (public) |
| [`nadayusc/confhead`](https://huggingface.co/nadayusc/confhead) | `run1`, the first trainval run, on the partial cache |
| [`nathancbao/confhead-charts`](https://huggingface.co/spaces/nathancbao/confhead-charts) | Trackio loss curves of the stage 0 and 1 runs on the partial cache, a snapshot from 2026-11-15 without `run1` |

## Run Catalogue

Every run trained on the partial `nuscenes-trainval` cache (fingerprint `99d44ff8c71c5355`, 110,929 train images) for 8 epochs, and each folder holds `best.pt`, `last.pt`, `config.json`, and `history.json`. The dev loss only shows that a run learned. AUSE and AURG (the areas under the sparsification-error and random-gain curves), which the Eval team scores on the val split, decide which run is best. Two runs with the same settings differ by 0.0004, so treat a gap below about 0.001 as a tie.

| Run | What changes from the baseline | Inputs (channels) | Best dev loss | Checkpoint |
|---|---|---:|---:|---|
| `run1` | nothing: the first trainval run | 68 | 0.5562 | [`nadayusc/confhead/run1`](https://huggingface.co/nadayusc/confhead/tree/main/run1) |
| `run-v1-radar` | nothing: the baseline, a repeat of `run1` | 68 | 0.5566 | [`csci566-depthconf/confhead/run-v1-radar`](https://huggingface.co/csci566-depthconf/confhead/tree/main/run-v1-radar) |
| `run-v0-no-radar` | no radar | 66 | 0.5562 | [`csci566-depthconf/confhead/run-v0-no-radar`](https://huggingface.co/csci566-depthconf/confhead/tree/main/run-v0-no-radar) |
| `m1-no-depth` | no depth channel | 67 | 0.5574 | [`csci566-depthconf/confhead/m1-no-depth`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-no-depth) |
| `m1-no-temporal` | no temporal spread | 67 | 0.5629 | [`csci566-depthconf/confhead/m1-no-temporal`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-no-temporal) |
| `m1-features-only` | decoder features alone | 64 | 0.5646 | [`csci566-depthconf/confhead/m1-features-only`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-features-only) |
| `m1-no-features` | no decoder features | 4 | 0.6167 | [`csci566-depthconf/confhead/m1-no-features`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-no-features) |
| `m1-depth-only` | depth channel alone | 1 | 0.6394 | [`csci566-depthconf/confhead/m1-depth-only`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-depth-only) |
| `m1-temporal-only` | temporal spread alone | 1 | 0.6778 | [`csci566-depthconf/confhead/m1-temporal-only`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-temporal-only) |
| `m1-seed1` | seed 1, which also picks other dev scenes | 68 | 0.5885 | [`csci566-depthconf/confhead/m1-seed1`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m1-seed1) |
| `m2-radar-lidar-fit` | radar residual from the lidar fit | 68 | 0.5507 | [`csci566-depthconf/confhead/m2-radar-lidar-fit`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m2-radar-lidar-fit) |
| `m2-radius0` | radar radius 0 | 68 | 0.5554 | [`csci566-depthconf/confhead/m2-radius0`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m2-radius0) |
| `m2-radius2` | radar radius 2 | 68 | 0.5548 | [`csci566-depthconf/confhead/m2-radius2`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m2-radius2) |
| `m2-sweeps1` | the keyframe radar sweep only | 68 | 0.5549 | [`csci566-depthconf/confhead/m2-sweeps1`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m2-sweeps1) |
| `m3-tau005` | `tau` 0.05, so other labels | 68 | 0.4261 | [`csci566-depthconf/confhead/m3-tau005`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m3-tau005) |
| `m3-tau02` | `tau` 0.2, so other labels | 68 | 0.6444 | [`csci566-depthconf/confhead/m3-tau02`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m3-tau02) |
| `m6-maxdepth50` | label depth range up to 50 m (other labels) | 68 | 0.5607 | [`csci566-depthconf/confhead/m6-maxdepth50`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m6-maxdepth50) |
| `m6-quarter-data` | 27,700 train images per epoch, a quarter | 68 | 0.5651 | [`csci566-depthconf/confhead/m6-quarter-data`](https://huggingface.co/csci566-depthconf/confhead/tree/main/m6-quarter-data) |

`run1` and `run-v1-radar` share every setting and the dev scenes. Their configs differ only in who trained them, the run folder, and the git commit, and the code in `src/` is the same at both commits. They are the one repeat pair the team has, and the 0.0004 noise figure comes from them.

Do not compare the dev loss of a `tau` run or of `m6-maxdepth50` with the others: they use other labels. Stages 4 and 5 of the run matrix (learning rate and head width, final seeds) have not run yet.

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

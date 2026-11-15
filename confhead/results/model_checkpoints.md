# Model Checkpoints

This page lists every training run of the confidence head and where its checkpoint lives. Start here to find a run to evaluate or compare. To see what each run tests and what the losses mean, read [the run matrix](../guides/run-matrix.md).

## Where Things Live

| Resource | What it holds |
|---|---|
| [`guides/run-matrix.md`](../guides/run-matrix.md) | The ablation matrix: every planned run, the flags it adds to [`configs/train-trainval.toml`](../configs/train-trainval.toml), its state, and the stage 1 findings |
| [`nathancbao/confhead`](https://huggingface.co/nathancbao/confhead) | The checkpoints of the ablation runs. Its model card lists the settings and per-epoch losses of each run |
| [`nathancbao/confhead` `EVAL_GUIDE.md`](https://huggingface.co/nathancbao/confhead/blob/main/EVAL_GUIDE.md) | For the Eval team: which run is which, what to score first, and how to load a run |
| [`nadayusc/confhead`](https://huggingface.co/nadayusc/confhead) | `run1`, the first trainval run. The training job uploads here unless you pass another `--repo` (`REPO_ID` in [`hub.py`](../src/confhead/hub.py)) |
| [`nathancbao/confhead-charts`](https://huggingface.co/spaces/nathancbao/confhead-charts) | Trackio loss curves of the runs in `nathancbao/confhead`. It is a snapshot from 2026-11-15 and does not include `run1` |

## The Runs

Every run trained on the `nuscenes-trainval` cache (fingerprint `99d44ff8c71c5355`) for 8 epochs, and each folder holds `best.pt`, `last.pt`, `config.json`, and `history.json`. The dev loss is a health check, not a ranking: Eval's AUSE and AURG on the val split decide which run is best. Two runs with the same settings differ by 0.0004, so treat a gap below about 0.001 as a tie.

| Run | What changes from the baseline | Inputs (channels) | Best dev loss | Checkpoint |
|---|---|---:|---:|---|
| `run1` | nothing: the first trainval run | 68 | 0.5562 | [`nadayusc/confhead/run1`](https://huggingface.co/nadayusc/confhead/tree/main/run1) |
| `run-v1-radar` | nothing: the baseline, a repeat of `run1` | 68 | 0.5566 | [`nathancbao/confhead/run-v1-radar`](https://huggingface.co/nathancbao/confhead/tree/main/run-v1-radar) |
| `run-v0-no-radar` | no radar | 66 | 0.5562 | [`nathancbao/confhead/run-v0-no-radar`](https://huggingface.co/nathancbao/confhead/tree/main/run-v0-no-radar) |
| `m1-no-depth` | no depth channel | 67 | 0.5574 | [`nathancbao/confhead/m1-no-depth`](https://huggingface.co/nathancbao/confhead/tree/main/m1-no-depth) |
| `m1-no-temporal` | no temporal spread | 67 | 0.5629 | [`nathancbao/confhead/m1-no-temporal`](https://huggingface.co/nathancbao/confhead/tree/main/m1-no-temporal) |
| `m1-features-only` | decoder features alone | 64 | 0.5646 | [`nathancbao/confhead/m1-features-only`](https://huggingface.co/nathancbao/confhead/tree/main/m1-features-only) |
| `m1-no-features` | no decoder features | 4 | 0.6167 | [`nathancbao/confhead/m1-no-features`](https://huggingface.co/nathancbao/confhead/tree/main/m1-no-features) |
| `m1-depth-only` | depth channel alone | 1 | 0.6394 | [`nathancbao/confhead/m1-depth-only`](https://huggingface.co/nathancbao/confhead/tree/main/m1-depth-only) |
| `m1-temporal-only` | temporal spread alone | 1 | 0.6778 | [`nathancbao/confhead/m1-temporal-only`](https://huggingface.co/nathancbao/confhead/tree/main/m1-temporal-only) |
| `m1-seed1` | seed 1, which also picks other dev scenes | 68 | 0.5885 | [`nathancbao/confhead/m1-seed1`](https://huggingface.co/nathancbao/confhead/tree/main/m1-seed1) |

`run1` and `run-v1-radar` share every setting, the training code, and the dev scenes; only the person and the run folder differ. They are the one repeat pair the team has, and the 0.0004 noise figure comes from them.

Stages 2 to 5 of the matrix (radar settings, `tau`, learning rate and head width, final seeds) have not run yet.

## Load a Checkpoint

```sh
hf download nathancbao/confhead m1-no-temporal/best.pt --local-dir checkpoints
```

```python
from confhead import load_head

head, inputs, info = load_head("checkpoints/m1-no-temporal/best.pt")
```

The runs take different input channels, so build each run's input from the `inputs` that `load_head` returns. [`notebooks/eval-template.ipynb`](../notebooks/eval-template.ipynb) shows the steps.

## Add a Run

1. Train it with the flags of its row in the run matrix and `--repo nathancbao/confhead`, as [the run matrix](../guides/run-matrix.md#how-to-run-one) shows, so that all ablation runs stay in one repository. The upload needs a login with write access to that repository.
2. Add a row to the table above, and fill in the state and dev loss in the run matrix.

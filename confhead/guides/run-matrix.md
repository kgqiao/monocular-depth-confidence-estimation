# Training Run Matrix

This table lists the training runs, what each one changes from the default config, and where each stands. Every run changes one thing from [`configs/train-trainval.toml`](../configs/train-trainval.toml). It is not a full grid. The runs and their settings are defined in [`configs/run-matrix.toml`](../configs/run-matrix.toml), which `confhead train` reads; this page explains them. Settings and their meaning are in [runs-and-configs.md](runs-and-configs.md).

> [!IMPORTANT]
> Every dev loss in the table is from the complete cache (204,894 entries, 152,124 train images), trained on 2026-11-17. The earlier runs on the partial cache had the same names. Their files are in the repository under the tag `cache-149k`, and their losses cannot be compared with the ones here.

## How to Run Them

[training-on-carc.md](training-on-carc.md) has the steps. In short, on a CARC login node:

```sh
uv run confhead train preview configs/run-matrix.toml
uv run confhead train run configs/run-matrix.toml
```

`--stage 2 3` or `--runs 'm2-*'` takes a part of the matrix. A run that is done and in the repository is skipped. The runs go to [`csci566-depthconf/confhead`](https://huggingface.co/csci566-depthconf/confhead). To add a run, add a `[[run]]` to the manifest and a row here.

## The Runs

States: `done`, `blocked` (waits on an earlier stage, and is not in the manifest yet). `confhead train status configs/run-matrix.toml` prints the live state of each run. Eval's AUSE and AURG come from the Eval team, on the official val split.

| Stage | Run | Question | `set` in the manifest | State | Best dev loss | AUSE | AURG |
|---|---|---|---|---|---|---|---|
| 0 | `run-v1-radar` | all inputs (the baseline) | none | done | 0.5589 | | |
| 0 | `run-v1-radar-repeat` | the baseline again: how much do two runs with the same settings differ? | none | done | 0.5581 | |
| 0 | `run-v0-no-radar` | no radar | `inputs.radar=false` | done | 0.5580 | | |
| 0b | radar coverage | how much of each image does radar cover? | no training: `examples/radar_coverage.py` | done | | | |
| 1 | `m1-no-temporal` | what does the temporal spread add? | `inputs.temporal=false` | done | 0.5648 | | |
| 1 | `m1-no-depth` | what does the depth channel add? | `inputs.depth=false` | done | 0.5607 | | |
| 1 | `m1-no-features` | what do the decoder features add? | `inputs.features=false` | done | 0.6129 | | |
| 1 | `m1-features-only` | features without the other groups | `inputs.depth=false inputs.temporal=false inputs.radar=false` | done | 0.5683 | | |
| 1 | `m1-temporal-only` | does the head beat the temporal cue alone? | `inputs.features=false inputs.depth=false inputs.radar=false` | done | 0.6713 | | |
| 1 | `m1-depth-only` | does the head beat the depth channel alone? | `inputs.features=false inputs.temporal=false inputs.radar=false` | done | 0.6368 | | |
| 1 | `m1-seed1` | how big is the seed noise? | `train.seed=1` | done | 0.5622 (other dev scenes) | | |
| 2 | `m2-radar-lidar-fit` | does the lidar fit change the radar residual? | `inputs.radar_cfg.align=lidar` | done | 0.5534 | | |
| 2 | `m2-radius0` | radar radius 0 | `inputs.radar_cfg.radius=0` | done | 0.5591 | | |
| 2 | `m2-radius2` | radar radius 2 | `inputs.radar_cfg.radius=2` | done | 0.5581 | | |
| 2 | `m2-sweeps1` | the keyframe radar sweep only | `inputs.radar_cfg.max_sweeps=1` | done | 0.5582 | | |
| 3 | `m3-tau005` | tau 0.05 | `inputs.labels.tau=0.05` | done | 0.4362 (other labels) | | |
| 3 | `m3-tau02` | tau 0.2 | `inputs.labels.tau=0.2` | done | 0.6349 (other labels) | | |
| 4 | `m4-lr3e-3` | learning rate 0.003 | `train.lr=0.003` | blocked | | | |
| 4 | `m4-lr3e-4` | learning rate 0.0003 | `train.lr=0.0003` | blocked | | | |
| 4 | `m4-epochs16` | 16 epochs | `train.epochs=16` | blocked | | | |
| 4 | `m4-wide` | a wider head | `head.widths=[128, 128, 64]` | blocked | | | |
| 4 | `m4-narrow` | a narrower head | `head.widths=[32, 32, 16]` | blocked | | | |
| 5 | `m5-seed1` | final config, seed 1 | `train.seed=1` on the best config | blocked | | | |
| 5 | `m5-seed2` | final config, seed 2 | `train.seed=2` on the best config | blocked | | | |
| 6 | `m6-maxdepth50` | labels only up to 50 m | `inputs.labels.max_depth=50.0` | done | 0.5633 (other labels) | | |
| 6 | `m6-quarter-data` | how much does the head lose with a quarter of the data? | `train.samples_per_epoch=38000` | done | 0.5664 | | |

Radar coverage (0b), measured on the 149,410 entries of the partial cache, training settings (`radius` 1, every sweep): the radar validity channel marks 3.4% of the grid cells (6.2% at radius 2, 1.7% with the keyframe sweep only). Only 4.8% of the lidar-labeled cells have a valid radar cell on them. The radar fit fails on 17.0% of images (32% on the two back-side cameras), and those images get an all-zero radar residual and mask. So the dev loss, which is computed on labeled cells, can barely move with radar. Stage 2 is still worth a few runs: start with `m2-radar-lidar-fit`, which removes the fit failures, then `m2-radius2`. Compare the loss on the labeled cells that have radar, not the mean dev loss.

Stages 0 and 1: all runs trained 8 of 8 epochs. This is dev loss only; Eval's scores decide. Every difference below is against `run-v1-radar` (0.5589, all inputs). The repeat noise is known from one pair only: `run-v1-radar` and `run-v1-radar-repeat` have the same settings and differ by 0.0008. The best epoch is the last or the one before it in every run, so the loss had nearly stopped falling.

- The decoder features carry most of the signal. Without them the loss is 0.6129 (+0.0540). The features alone (0.5683, with radar off too) are 0.0094 worse than all inputs.
- The temporal spread helps a little: removing it costs +0.0059.
- The depth channel adds little: removing it costs +0.0018, about twice the one repeat gap.
- Radar made no difference on dev loss: `run-v0-no-radar` 0.5580 against 0.5589, a gap of about the repeat noise.
- One channel alone (0.6368 depth, 0.6713 temporal) is far worse than any run with the features.
- `m1-seed1` (0.5622) is not comparable on dev loss, because the seed also picks other dev scenes. Its use is Eval's score on val.

Stages 2 and 3 (all six runs trained 8 of 8 epochs; dev loss only, Eval's scores decide): `m2-radar-lidar-fit` (0.5534) is 0.0055 under `run-v1-radar` (0.5589), the largest gap under the baseline of any run. `m2-radius0`, `m2-radius2` and `m2-sweeps1` are within 0.0008 of the baseline (0.5591, 0.5581, 0.5582), which is no more than the repeat noise. The two `tau` runs use other labels, so their dev loss (0.4362 at 0.05, 0.6349 at 0.2) cannot be compared with the others. Stage 6: `m6-quarter-data` (38,000 images per epoch) reaches 0.5664, 0.0075 above the baseline. `m6-maxdepth50` uses other labels (0.5633) and also cannot be compared.

The runs and their files are in [`csci566-depthconf/confhead`](https://huggingface.co/csci566-depthconf/confhead) on Hugging Face. The repository is gated: log in, and ask for access (or be added by an admin) before a download works. Its `EVAL_GUIDE.md` tells the Eval team which run is which and what to score first; [`results/EVAL_GUIDE.md`](../results/EVAL_GUIDE.md) is a copy. Both still give the dev losses of the partial-cache runs.

Stage 4 runs on the best inputs of stages 1 to 3. Stage 5 uses the best config of all stages. Their rows go into the manifest when those configs are chosen. The `head.widths` lists of stage 4 are written as `head.widths=[128, 128, 64]` in `set`.

## See the Runs in Trackio

[`examples/log_runs_to_trackio.py`](../examples/log_runs_to_trackio.py) replays the `history.json` of finished runs into [trackio](https://huggingface.co/docs/trackio), so that the loss curves of all runs draw on the same charts, and the settings of each run can group and filter them. It does not change training. A run that is already in the project is skipped.

The charts of the stage 0 and 1 runs on the partial cache are in a static Space: [`nathancbao/confhead-charts`](https://huggingface.co/spaces/nathancbao/confhead-charts) (public, read only). It is a snapshot from 2026-11-15: after the runs on the complete cache finish, replay them and sync again (below).

What works from a free Hugging Face account: the replay into the local store, and a static Space made from that store. What does not: a live trackio Space (`--space-id`). trackio deploys a Gradio Space for that, and on a free account Hugging Face answered `402 Payment Required`. Its error message says that hosting Gradio and Docker Spaces on free hardware needs a PRO subscription and that static Spaces are free; we did not test a PRO account. Static Spaces cannot be private.

```sh
# 1. replay run folders, or every run of a Hugging Face model repository, into the local store
uv run --with trackio examples/log_runs_to_trackio.py /scratch1/$USER/confhead-runs/m1-seed1
uv run --with trackio examples/log_runs_to_trackio.py --hf-repo csci566-depthconf/confhead

# 2. open the local dashboard
uv run --with trackio trackio show --project confhead

# 3. refresh the static Space from the local store (needs a login with write access to the Space)
uv run --with "trackio[spaces]" python -c "import trackio; trackio.sync(project='confhead', space_id='<user>/<space>', sdk='static', force=True)"
```

## Reading the Table

- The best dev loss is a health check, not a ranking. Eval's AUSE and AURG decide which config wins. A gap smaller than the difference between `run-v1-radar` and `run-v1-radar-repeat` (0.0008) is a tie.
- Do not compare dev loss across `tau`: another `tau` makes other labels.
- The seed also picks the dev scenes (`dev_scenes` in `train.py`), so compare seeds on Eval's scores, not on dev loss.
- `train.epochs` changes the cosine learning-rate schedule, so 16 epochs is a different run, not 8 plus more.
- A run took 0.8 to 3.1 hours of training on the complete cache, and a GPU job can wait many hours in the queue.

## Not in the Table

These need a code change or a new cache: class weight off or another loss, more weight on night and rain images, the metric Depth Anything V2 checkpoint, another tap layer, other temporal offsets, and another number of radar sweeps in the cache. `inputs.labels.align=radar`, `train.weight_decay`, and `train.batch_size` are left out on purpose.

# Training Run Matrix

This table lists the training runs we plan, what each one changes from the default config, and where each stands. Every run changes one thing from [`configs/train-trainval.toml`](../configs/train-trainval.toml). It is not a full grid. Settings and their meaning are in [runs-and-configs.md](runs-and-configs.md). The steps to run a job are in [training-on-carc.md](training-on-carc.md).

## How to Run One

Add the flags of the row and name the run with `output.name`:

```sh
sbatch examples/carc-train.sbatch configs/train-trainval.toml --repo csci566-depthconf/confhead --set inputs.temporal=false --set output.name=m1-no-temporal
```

`--repo` sends the uploads to [`csci566-depthconf/confhead`](https://huggingface.co/csci566-depthconf/confhead), where the other ablation runs are; it needs a login with write access to that repository. Without `--repo`, the job uploads to `nadayusc/confhead`. To upload a finished run by hand, use [`examples/upload_run.py`](../examples/upload_run.py) with the same `--repo`. Then fill in the row here and add the run to [`results/model_checkpoints.md`](../results/model_checkpoints.md).

## The Runs

States: `done`, `running`, `submitted` (job sent, result not read yet), `todo`, `blocked` (waits on an earlier stage). Eval's AUSE and AURG come from the Eval team, on the official val split.

| Stage | Run | Question | Flags added to the default config | State | Best dev loss | AUSE | AURG |
|---|---|---|---|---|---|---|---|
| 0 | `run-v0-no-radar` | no radar | `--set inputs.radar=false` | done | 0.5562 | | |
| 0 | `run-v1-radar` | all inputs (the baseline) | none | done | 0.5566 | | |
| 0b | radar coverage | how much of each image does radar cover? | no training: `examples/radar_coverage.py` | done | | | |
| 1 | `m1-no-temporal` | what does the temporal spread add? | `--set inputs.temporal=false` | done | 0.5629 | | |
| 1 | `m1-no-depth` | what does the depth channel add? | `--set inputs.depth=false` | done | 0.5574 | | |
| 1 | `m1-no-features` | what do the decoder features add? | `--set inputs.features=false` | done | 0.6167 | | |
| 1 | `m1-features-only` | features without the other groups | `--set inputs.depth=false --set inputs.temporal=false --set inputs.radar=false` | done | 0.5646 | | |
| 1 | `m1-temporal-only` | does the head beat the temporal cue alone? | `--set inputs.features=false --set inputs.depth=false --set inputs.radar=false` | done | 0.6778 | | |
| 1 | `m1-depth-only` | does the head beat the depth channel alone? | `--set inputs.features=false --set inputs.temporal=false --set inputs.radar=false` | done | 0.6394 | | |
| 1 | `m1-seed1` | how big is the seed noise? | `--set train.seed=1` | done | 0.5885 (other dev scenes) | | |
| 2 | `m2-radar-lidar-fit` | does the lidar fit change the radar residual? | `--set inputs.radar_cfg.align=lidar` | done | 0.5507 | | |
| 2 | `m2-radius0` | radar radius 0 | `--set inputs.radar_cfg.radius=0` | done | 0.5554 | | |
| 2 | `m2-radius2` | radar radius 2 | `--set inputs.radar_cfg.radius=2` | done | 0.5548 | | |
| 2 | `m2-sweeps1` | the keyframe radar sweep only | `--set inputs.radar_cfg.max_sweeps=1` | done | 0.5549 | | |
| 3 | `m3-tau005` | tau 0.05 | `--set inputs.labels.tau=0.05` | done | 0.4261 | | |
| 3 | `m3-tau02` | tau 0.2 | `--set inputs.labels.tau=0.2` | done | 0.6444 | | |
| 4 | `m4-lr3e-3` | learning rate 0.003 | `--set train.lr=0.003` | blocked | | | |
| 4 | `m4-lr3e-4` | learning rate 0.0003 | `--set train.lr=0.0003` | blocked | | | |
| 4 | `m4-epochs16` | 16 epochs | `--set train.epochs=16` | blocked | | | |
| 4 | `m4-wide` | a wider head | `--set head.widths=[128,128,64]` (check that the list syntax parses) | blocked | | | |
| 4 | `m4-narrow` | a narrower head | `--set head.widths=[32,32,16]` (check that the list syntax parses) | blocked | | | |
| 5 | `m5-seed1` | final config, seed 1 | `--set train.seed=1` on the best config | blocked | | | |
| 5 | `m5-seed2` | final config, seed 2 | `--set train.seed=2` on the best config | blocked | | | |
| 6 | `m6-quarter-data` | how much does the head lose with a quarter of the data? | `--set train.samples_per_epoch=27700` (about a quarter of the 110,929 train images) | done | 0.5651 | | |
| 6 | `m6-maxdepth50` | labels only up to 50 m | `--set inputs.labels.max_depth=50` | done | 0.5607 (other labels) | | |

Radar coverage (0b), all 149,410 cache entries, training settings (`radius` 1, every sweep): the radar validity channel marks 3.4% of the grid cells (6.2% at radius 2, 1.7% with the keyframe sweep only). Only 4.8% of the lidar-labeled cells have a valid radar cell on them. The radar fit fails on 17.0% of images (32% on the two back-side cameras), and those images get an all-zero radar residual and mask. So the dev loss, which is computed on labeled cells, can barely move with radar. Stage 2 is still worth a few runs: start with `m2-radar-lidar-fit`, which removes the fit failures, then `m2-radius2`. Compare the loss on the labeled cells that have radar, not the mean dev loss.

Stage 1: all seven runs trained 8 of 8 epochs. This is dev loss only; Eval's scores decide. Every difference below is against `run-v1-radar` (0.5566, all inputs). The repeat noise is known from one pair only: the same settings trained twice (`run1` in `nadayusc/confhead`, and `run-v1-radar`) differ by 0.0004. In every run the best epoch is the last, so the loss was still falling.

- The decoder features carry most of the signal. Without them the loss is 0.6167 (+0.0601). The features alone (0.5646, with radar off too) are 0.0080 worse than all inputs.
- The temporal spread helps a little: removing it costs +0.0063.
- The depth channel is close to noise: removing it costs +0.0008, about twice the one repeat gap.
- Radar made no difference on dev loss: `run-v0-no-radar` 0.5562 against 0.5566.
- One channel alone (0.6394 depth, 0.6778 temporal) is far worse than any run with the features.
- `m1-seed1` (0.5885) is not comparable on dev loss, because the seed also picks other dev scenes. Its use is Eval's score on val.

The runs and their files are in [`csci566-depthconf/confhead`](https://huggingface.co/csci566-depthconf/confhead) on Hugging Face (public). Its `EVAL_GUIDE.md` tells the Eval team which run is which and what to score first.

Stages 2 and 3 (all six runs trained 8 of 8 epochs; dev loss only, Eval's scores decide): `m2-radar-lidar-fit` (0.5507) is 0.0059 under `run-v1-radar` (0.5566), the largest gap of any run against the baseline, and `m2-radius0`, `m2-radius2` and `m2-sweeps1` are within 0.0018 of it (0.5554, 0.5548, 0.5549). The two `tau` runs use other labels, so their dev loss (0.4261 at 0.05, 0.6444 at 0.2) cannot be compared with the others. Stage 6: `m6-quarter-data` (27,700 images per epoch) reaches 0.5651, 0.0085 above the baseline. `m6-maxdepth50` uses other labels (0.5607) and also cannot be compared. Hugging Face holds all of these runs.

Stage 4 runs on the best inputs of stages 1 to 3. Stage 5 uses the best config of all stages.

## See the Runs in Trackio

[`examples/log_runs_to_trackio.py`](../examples/log_runs_to_trackio.py) replays the `history.json` of finished runs into [trackio](https://huggingface.co/docs/trackio), so that the loss curves of all runs draw on the same charts, and the settings of each run can group and filter them. It does not change training. A run that is already in the project is skipped.

The charts of all runs are in a static Space: [`nathancbao/confhead-charts`](https://huggingface.co/spaces/nathancbao/confhead-charts) (public, read only). It is a snapshot: after new runs finish, replay them and sync again (below).

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

- The best dev loss is a health check, not a ranking. Eval's AUSE and AURG decide which config wins. A gap smaller than the difference between `run1` and `run-v1-radar` (0.0004) is a tie.
- Do not compare dev loss across `tau`: another `tau` makes other labels.
- The seed also picks the dev scenes (`dev_scenes` in `train.py`), so compare seeds on Eval's scores, not on dev loss.
- `train.epochs` changes the cosine learning-rate schedule, so 16 epochs is a different run, not 8 plus more.
- A run took 41 minutes to 5.5 hours so far, depending on the node, and a GPU job can wait many hours in the queue. Do not start two `uv run` jobs together: they share one `.venv`, and the second rebuilds it under the first. Start one, wait until it runs, then start the next.

## Not in the Table

These need a code change or a new cache: class weight off or another loss, more weight on night and rain images, the metric Depth Anything V2 checkpoint, another tap layer, other temporal offsets, and another number of radar sweeps in the cache. `inputs.labels.align=radar`, `train.weight_decay`, and `train.batch_size` are left out on purpose.

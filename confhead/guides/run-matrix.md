# Training Run Matrix

This table lists the training runs we plan, what each one changes from the default config, and where each stands. Every run changes one thing from [`configs/train-trainval.toml`](../configs/train-trainval.toml). It is not a full grid. Settings and their meaning are in [runs-and-configs.md](runs-and-configs.md). The steps to run a job are in [training-on-carc.md](training-on-carc.md).

## How to Run One

Add the flags of the row and name the run with `output.name`:

```sh
sbatch examples/carc-train.sbatch configs/train-trainval.toml --set inputs.temporal=false --set output.name=m1-no-temporal
```

Then upload it with [`examples/upload_run.py`](../examples/upload_run.py) and fill in the row.

## The Runs

States: `done`, `running`, `submitted` (job sent, result not read yet), `todo`, `blocked` (waits on an earlier stage). Owner is filled in when a run is taken. Eval's AUSE and AURG come from the Eval team, on the official val split.

| Stage | Run | Question | Flags added to the default config | Owner | State | Best dev loss | AUSE | AURG |
|---|---|---|---|---|---|---|---|---|
| 0 | `run1` | all inputs | none | Nathan A | done | 0.5562 | | |
| 0 | `run-v0-no-radar` | no radar | `--set inputs.radar=false` | Nathan B | done | 0.5562 | | |
| 0 | `run-v1-radar` | repeat of `run1` | none | Nathan B | done | 0.5566 | | |
| 0b | radar coverage | how much of each image does radar cover? | no training: `examples/radar_coverage.py` | Nathan B | done | | | |
| 1 | `m1-no-temporal` | what does the temporal spread add? | `--set inputs.temporal=false` | | todo | | | |
| 1 | `m1-no-depth` | what does the depth channel add? | `--set inputs.depth=false` | | todo | | | |
| 1 | `m1-no-features` | what do the decoder features add? | `--set inputs.features=false` | | todo | | | |
| 1 | `m1-features-only` | features without the other groups | `--set inputs.depth=false --set inputs.temporal=false --set inputs.radar=false` | | todo | | | |
| 1 | `m1-temporal-only` | does the head beat the temporal cue alone? | `--set inputs.features=false --set inputs.depth=false --set inputs.radar=false` (check that it runs) | | todo | | | |
| 1 | `m1-depth-only` | does the head beat the depth channel alone? | `--set inputs.features=false --set inputs.temporal=false --set inputs.radar=false` (check that it runs) | | todo | | | |
| 1 | `m1-seed1` | how big is the seed noise? | `--set train.seed=1` | | todo | | | |
| 2 | `m2-radar-lidar-fit` | does the lidar fit change the radar residual? | `--set inputs.radar_cfg.align=lidar` | | todo | | | |
| 2 | `m2-radius0` | radar radius 0 | `--set inputs.radar_cfg.radius=0` | | todo | | | |
| 2 | `m2-radius2` | radar radius 2 | `--set inputs.radar_cfg.radius=2` | | todo | | | |
| 2 | `m2-sweeps1` | the keyframe radar sweep only | `--set inputs.radar_cfg.max_sweeps=1` | | todo | | | |
| 3 | `m3-tau005` | tau 0.05 | `--set inputs.labels.tau=0.05` | | todo | | | |
| 3 | `m3-tau02` | tau 0.2 | `--set inputs.labels.tau=0.2` | | todo | | | |
| 4 | `m4-lr3e-3` | learning rate 0.003 | `--set train.lr=0.003` | | blocked | | | |
| 4 | `m4-lr3e-4` | learning rate 0.0003 | `--set train.lr=0.0003` | | blocked | | | |
| 4 | `m4-epochs16` | 16 epochs | `--set train.epochs=16` | | blocked | | | |
| 4 | `m4-wide` | a wider head | `--set head.widths=[128,128,64]` (check that the list syntax parses) | | blocked | | | |
| 4 | `m4-narrow` | a narrower head | `--set head.widths=[32,32,16]` (check that the list syntax parses) | | blocked | | | |
| 5 | `m5-seed1` | final config, seed 1 | `--set train.seed=1` on the best config | | blocked | | | |
| 5 | `m5-seed2` | final config, seed 2 | `--set train.seed=2` on the best config | | blocked | | | |

Radar coverage (0b), all 149,410 cache entries, training settings (`radius` 1, every sweep): the radar validity channel marks 3.4% of the grid cells (6.2% at radius 2, 1.7% with the keyframe sweep only). Only 4.8% of the lidar-labeled cells have a valid radar cell on them. The radar fit fails on 17.0% of images (32% on the two back-side cameras), and those images get an all-zero radar residual and mask. So the dev loss, which is computed on labeled cells, can barely move with radar. Stage 2 is still worth a few runs: start with `m2-radar-lidar-fit`, which removes the fit failures, then `m2-radius2`. Compare the loss on the labeled cells that have radar, not the mean dev loss.

Stage 4 runs on the best inputs of stages 1 to 3. Stage 5 uses the best config of all stages.

## Reading the Table

- The best dev loss is a health check, not a ranking. Eval's AUSE and AURG decide which config wins. A gap smaller than the difference between `run1` and `run-v1-radar` (0.0004) is a tie.
- Do not compare dev loss across `tau`: another `tau` makes other labels.
- The seed also picks the dev scenes (`dev_scenes` in `train.py`), so compare seeds on Eval's scores, not on dev loss.
- `train.epochs` changes the cosine learning-rate schedule, so 16 epochs is a different run, not 8 plus more.
- A run takes 3 to 5.5 hours. Do not start two `uv run` jobs together: they share one `.venv`, and the second rebuilds it under the first. Start one, wait until it runs, then start the next.

## Not in the Table

These need a code change or a new cache: class weight off or another loss, more weight on night and rain images, the metric Depth Anything V2 checkpoint, another tap layer, other temporal offsets, and another number of radar sweeps in the cache. `inputs.labels.align=radar`, `train.weight_decay`, and `train.batch_size` are left out on purpose.

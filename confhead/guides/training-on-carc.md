# Train the Head on CARC

This guide trains the confidence head on the v1.0-trainval cache, as a batch job on one CARC GPU. The job needs no open session. It writes its progress to a log, and a run that stops at its time limit continues when you submit it again.

[`notebooks/train-head.ipynb`](../notebooks/train-head.ipynb) runs the same training step by step. Both call `confhead.train`, so the notebook is the place to try a setting on the mini cache before you submit a long job.

## What You Need

- The trainval cache, built with [cache-building-on-carc.md](cache-building-on-carc.md). The config expects it in `/scratch1/$USER/confhead-cache/nuscenes-trainval`.
- The repository cloned on CARC, and uv, as in the cache guide.
- `SBATCH_ACCOUNT` set to your CARC project account, as in step 1 of the cache guide.

Run every command on a CARC login node, from the `confhead` folder:

```sh
cd ~/Vision-Project-CSCI-566/confhead
git pull
uv sync
mkdir -p logs
```

## 1. Look at the Config

[`configs/train-trainval.toml`](../configs/train-trainval.toml) holds every setting of the run. These are the settings you are most likely to change:

| Setting | Value | What it is |
|---|---|---|
| `data.cache` | `/scratch1/$USER/confhead-cache/nuscenes-trainval` | The cache to train on |
| `output.runs` | `/scratch1/$USER/confhead-runs` | The folder of all runs |
| `output.name` | `run1` | The run folder inside `output.runs` |
| `inputs.features`, `inputs.depth`, `inputs.temporal`, `inputs.radar` | `true` | The channel groups of `x`. Turn one off for an ablation. |
| `train.samples_per_epoch` | `0` | Train images per epoch, drawn at random. 0 means all of them. |

To train on a cache that a teammate built, point `data.cache` to their folder, for example `/scratch1/<user>/confhead-cache/nuscenes-trainval`.

> [!TIP]
> To change a setting for one run only, add `--set section.key=value` to the command. A nested setting takes more dots: `--set inputs.labels.tau=0.2`. [`src/confhead/train.py`](../src/confhead/train.py) lists every setting with its default.

## 2. Estimate the Run

This job trains nothing. It reads the splits, measures the input statistics, and times the loading and the training step:

```sh
sbatch --time=01:00:00 examples/carc-train.sbatch configs/train-trainval.toml --estimate
```

`sbatch` prints the job number: `Submitted batch job <job>`. When the job ends, read its log:

```sh
cat logs/confhead-train-<job>.out
```

The last lines look like this:

```
labels   ...% of cells labeled; positive rate ...% (gate 5% to 70%: inside); class weight ...
speed    loading ... images/s, training step ... images/s
epoch    ... images in about ... min; 8 epochs in about ... h
```

> [!IMPORTANT]
> The `labels` line must say `inside`. A positive rate outside the gate means that the depth needs a scale per camera before training, as the Run 1 plan says. Stop there and bring the numbers to the team.

If one epoch takes more than 2 hours, set `train.samples_per_epoch` in the config to half the train images. The `train` line of the log gives their number.

## 3. Submit the Training

```sh
sbatch examples/carc-train.sbatch configs/train-trainval.toml
```

The job has 12 hours, set by `#SBATCH --time` in [`examples/carc-train.sbatch`](../examples/carc-train.sbatch). If the estimate is longer, the job stops at the limit and step 5 continues it. You can also submit with a longer `--time`, up to the limit of the partition.

## 4. Watch the Job

```sh
squeue --me
tail -f logs/confhead-train-<job>.out
```

The log prints a line five times per epoch, with the loss, the speed, and the time left in the epoch. After each epoch it prints the train and dev loss:

```
epoch 1: train loss ..., dev loss ..., ... min, best so far
```

To stop the job, use `scancel <job>`.

## 5. Continue a Run That Stopped

If the job reached its time limit or failed, submit the same command again:

```sh
sbatch examples/carc-train.sbatch configs/train-trainval.toml
```

The run folder holds `last.pt` after every epoch, so the job continues after the last epoch that finished, and loses at most one epoch. The batch order of an epoch depends only on the seed and the epoch number, so a continued run trains as if it had never stopped. When every epoch is done, the log says `... already holds 8 of 8 epochs; nothing to train`.

A run folder refuses other settings. If you changed a setting in between, the log names it, for example `train.lr 0.001 -> 0.002`. Undo the change, or give the run a new `output.name`.

## 6. Find the Checkpoint

The last line of the log names the checkpoint for the eval team:

```
best     epoch <n> of 8, dev loss <loss>: /scratch1/<user>/confhead-runs/run1/best.pt
```

| File in the run folder | What it holds |
|---|---|
| `config.json` | every setting, the dev scenes, the class weight, the cache fingerprint, the git commit |
| `best.pt` | the head of the epoch with the lowest dev loss |
| `last.pt` | the head of the last epoch, with the optimizer and the schedule, to continue the run |
| `history.json` | the train and dev loss of each epoch |

[`notebooks/eval-template.ipynb`](../notebooks/eval-template.ipynb) loads `best.pt` with `load_head` and predicts `p_error` for val images.

## Run an Ablation

Give each ablation its own `output.name`. Several jobs can run at once, each on its own GPU:

```sh
sbatch examples/carc-train.sbatch configs/train-trainval.toml --set inputs.radar=false --set output.name=run1-no-radar
sbatch examples/carc-train.sbatch configs/train-trainval.toml --set inputs.radar_cfg.align=lidar --set output.name=run1-radar-lidar-fit
```

These two runs are the radar ablation of action 3 in [`Docs/training-next-actions.md`](../../Docs/training-next-actions.md). The labels do not change with the inputs, so the eval team can compare the runs directly.

## If Something Goes Wrong

| Message | What to do |
|---|---|
| `... has no index.json` | `data.cache` does not point to a built cache. Check the path, or build the cache first. |
| `... is a run with other settings: ...` | Undo the change that the message names, or set another `output.name`. |
| `unknown setting ...` | The setting is spelled wrong. The message lists the settings of that section. |
| `torch ... has no kernels for <GPU>` | The installed torch does not support this GPU. Run `git pull` and `uv sync` on the login node, then submit again. |
| `... the class weight needs both classes` | The sample of train images holds no label above `tau`, or none below it. Check `inputs.labels.tau`. |

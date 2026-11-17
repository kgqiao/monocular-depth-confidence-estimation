# Train the Head on CARC

This guide trains the runs of the run matrix on CARC and puts them on Hugging Face. `confhead train` is the only way to train a head that the team uses: it gives every run the same cache, the same job, and the same checks. One command submits a Slurm job per run. The jobs run at the same time, need no open session, and upload each run when it ends.

[`notebooks/train-head.ipynb`](../notebooks/train-head.ipynb) shows the training step by step on the mini cache. It is a tutorial, and its runs do not go to the repository.

| Command | What it does |
|---|---|
| `confhead train preview <manifest>` | Checks the login, the cache, and each run, and prints what `run` would do. It trains nothing. |
| `confhead train run <manifest>` | Submits a job for each run that has epochs left, uploads the runs that are trained but not uploaded, and submits a report job. |
| `confhead train status <manifest>` | Prints where each run stands. |

The manifest is [`configs/run-matrix.toml`](../configs/run-matrix.toml). [run-matrix.md](run-matrix.md) says what each run tests.

## What You Need

- The complete trainval cache in `/scratch1/naday/confhead-cache/nuscenes-trainval`, with 204,894 entries. Every run reads this one cache, whoever submits the job. `preview` refuses a cache with another entry count.
- The repository cloned on CARC, and uv, as in [cache-building-on-carc.md](cache-building-on-carc.md).
- `SBATCH_ACCOUNT` set to your CARC project account, as in step 1 of the cache guide.
- A Hugging Face login that can write to [`csci566-depthconf/confhead`](https://huggingface.co/csci566-depthconf/confhead): a role of `write` or `admin` in the `csci566-depthconf` organization, and a write token. Log in once with `uv run hf auth login`.

Run every command on a CARC login node, from the `confhead` folder:

```sh
cd ~/Vision-Project-CSCI-566/confhead
git pull
uv sync
```

`run` refuses to submit when tracked files have changes, so that the `config.json` of a run names the code that trained it.

## 1. Preview

```sh
uv run confhead train preview configs/run-matrix.toml
```

To take a part of the matrix, name stages or runs. The same selection works for `run` and `status`:

```sh
uv run confhead train preview configs/run-matrix.toml --stage 2 3
uv run confhead train preview configs/run-matrix.toml --runs 'm1-*' run-v1-radar
```

The preview prints the login, the cache, and one line per run:

```
login     <user> can write to csci566-depthconf/confhead
cache     /scratch1/naday/confhead-cache/nuscenes-trainval: 204894 entries, fingerprint 99d44ff8c71c5355

run             stage  fingerprint       images/epoch  hours  action
run-v1-radar    0      ...               ...           ...    train; replaces the repository run
m1-no-temporal  1      ...               ...           ...    resume after epoch 5 of 8
m2-radius2      2      ...               ...                  done, in the repository

18 runs: ... GPU jobs, ... uploads only, ... done, ... in the queue
estimate  about ... GPU-hours in total, ...
ready     send the same arguments to `confhead train run`
```

A `problem` line names what stops `run`, and the preview ends without `ready`.

To prove that the login can write, the preview creates a branch `access-check-<user>` in the repository and deletes it at once. It changes nothing else.

The hours are a ballpark. For each run, the preview takes the speed of the earlier run of the same name in the repository and scales it to the new number of images. It also prints the total at 28 images per second, the slowest loading measured on CARC. Runs of one size took 0.2 to 5.5 hours before, by the node they ran on.

### The Action of Each Run

`preview`, `run`, and `status` read the state of a run from three places: its run folder, the repository, and the Slurm queue. No file records what was submitted.

| What is found | Action |
|---|---|
| No run folder | `train`: a new job |
| A run folder with the same fingerprint and epochs left | `resume after epoch <n>`: a job that continues from `last.pt` |
| A run folder with every epoch, and the run is not in the repository | `upload only`: `run` uploads it from the login node, with no job |
| The repository holds the run with the same fingerprint and every epoch | `done` |
| A job of this run is queued or running | `job <number>`: nothing is submitted |
| A run folder with another fingerprint | `stale run folder`: needs `--archive-stale` |
| The repository holds the name with another fingerprint | `replaces the repository run`: needs `--replace` |

### The Fingerprint

The fingerprint of a run is a hash of what makes it a model:

- every setting except the `data`, `output`, and `run` sections
- the fingerprint of the cache build
- the number of train images
- the dev scenes

Two runs with the same fingerprint are the same model, wherever they trained. The number of train images separates the complete cache from a copy that lacks entries, which has the same cache fingerprint. `config.json` saves the git commit, and the fingerprint leaves it out, so a commit that changes no training code does not make every run new.

## 2. Trial Run

Before the real runs, send one short run through the whole path:

```sh
uv run confhead train run configs/run-matrix.toml --runs run-v1-radar --trial
```

A trial trains 1 epoch of 10,000 images under the name `trial-run-v1-radar`, which takes about 6 minutes at the slowest rate. It uses the same job script, environment, checkpoints, upload, and report as a real run. When the report says the trial is done, delete it on disk and in the repository:

```sh
uv run confhead train clean-trials configs/run-matrix.toml
```

## 3. Run

```sh
uv run confhead train run configs/run-matrix.toml
```

`run` makes the checks of the preview again and submits nothing if one fails. Then it prints a line per job:

```
submitted run-v1-radar: job <job>, train, log logs/train/run-v1-radar-<job>.out
report    job <job> starts when they end: logs/train/report-<job>.out
```

Each job has 16 hours, the `time` of the manifest; `--time 24:00:00` sets another limit for one command. A job writes `last.pt` and `best.pt` to `/scratch1/$USER/confhead-runs/<run>` after every epoch, and uploads the run folder after every epoch and once more at the end.

Each job builds its own Python environment in the node's `$TMPDIR` (or `/tmp`) and deletes it when it ends. Jobs never use the shared `.venv`, so they do not disturb each other. The note in step 5 of [cache-building-on-carc.md](cache-building-on-carc.md) says why.

### Replace a Run

A run name is the folder of the run on disk and in the repository. When the repository holds another model under a name of the matrix, `run` stops unless you pass `--replace`:

```sh
uv run confhead train run configs/run-matrix.toml --replace --tag cache-149k
```

Before the first job starts, `run` sets a tag on the current commit of the repository, unless a tag names that commit already. `--tag` gives the tag its name; without it the name is `before-<date>-<time>`. The files of the replaced runs stay in the history of the repository:

```sh
uv run hf download csci566-depthconf/confhead m1-no-temporal/best.pt --revision cache-149k --local-dir .
```

A run folder on disk that holds another model under the same name stops `run` in the same way. `--archive-stale` renames the folder to `<run>.stale-<fingerprint>` and deletes nothing.

## 4. Watch the Jobs

```sh
squeue --me
tail -f logs/train/<run>-<job>.out
uv run confhead train status configs/run-matrix.toml
```

A log prints a line five times per epoch with the loss, the speed, and the time left in the epoch. After each epoch it prints the losses and the commit of the upload:

```
epoch 1: train loss ..., dev loss ..., ... min, best so far
uploaded epoch 1 of 8: https://huggingface.co/csci566-depthconf/confhead/commit/...
```

> [!IMPORTANT]
> The positive rate is the share of labeled cells with an error above `tau`. A log line `the positive rate is outside the gate of 5% to 70%` means that the depth needs a scale per camera before training. The [Run 1 plan](../../Docs/Wiki/CS566%20Confidence%20Head%20Training%20Run%201.md) sets this gate. Stop that run with `scancel <job>` and bring the numbers to the team.

To stop a job, use `scancel <job>`. The run keeps its checkpoints.

## 5. Read the Report

The report job starts when every training job of the command has ended, whether it finished or failed. It writes `logs/train/report-<job>.out`:

```
run             state                  epochs  best dev loss  job    Slurm      elapsed
run-v1-radar    done                   8 of 8  ...            <job>  COMPLETED  ...
m1-no-temporal  STOPPED                5 of 8  ...            <job>  TIMEOUT    ...
m2-radius2      TRAINED, NOT UPLOADED  8 of 8  ...            <job>  FAILED     ...

16 of 18 runs are done and in the repository.
```

For each run that is not done, the report gives the Slurm state, the exit code, and the last 15 lines of the job's log.

| State | What it means |
|---|---|
| `done` | Every epoch is trained and the run is in the repository. |
| `STOPPED` | The job ended with epochs left: a time limit, a node failure, or an error. |
| `TRAINED, NOT UPLOADED` | Every epoch is trained, and the last upload failed three times. The job exits with an error. |
| `NOT STARTED` | No checkpoint was written. |

## 6. Continue What Is Left

Read the cause in the report, fix it, and send the same `run` command again:

```sh
uv run confhead train run configs/run-matrix.toml
```

It uploads the runs that are trained, submits jobs for the runs with epochs left, and skips the runs that are done or still in the queue. A continued run loses at most one epoch. The batch order of an epoch depends only on the seed and the epoch number, so a continued run trains as if it had never stopped.

## 7. After the Runs

- Fill in the dev loss of each run in [run-matrix.md](run-matrix.md) and [`results/model_checkpoints.md`](../results/model_checkpoints.md).
- [tracking-runs.md](tracking-runs.md) draws the loss curves of the repository's runs.

Each upload rewrites the model card of the repository from the `config.json` and `history.json` of every run in it. When two jobs upload at once, the later commit is refused, reads the repository again, and retries, so neither run drops off the card.

To download a head:

```sh
uv run hf download csci566-depthconf/confhead m1-no-temporal/best.pt --local-dir .
```

| File in the run folder | What it holds |
|---|---|
| `config.json` | every setting, the dev scenes, the class weight, the cache fingerprint, the git commit, the run fingerprint |
| `best.pt` | the head of the epoch with the lowest dev loss |
| `last.pt` | the head of the last epoch, with the optimizer and the schedule, to continue the run |
| `history.json` | the train and dev loss of each epoch |
| `steps.json` | the mean train loss five times per epoch |

## Add a Run

Add a `[[run]]` to [`configs/run-matrix.toml`](../configs/run-matrix.toml) and a row to [run-matrix.md](run-matrix.md):

```toml
[[run]]
name = "m4-lr3e-3"
stage = 4
group = "optimizer"
question = "learning rate 0.003"
set = ["train.lr=0.003"]
```

`set` changes settings of [`configs/train-trainval.toml`](../configs/train-trainval.toml), written as `section.key=value`. A nested setting takes more dots: `inputs.labels.tau=0.2`. [`src/confhead/train.py`](../src/confhead/train.py) lists every setting with its default, and [runs-and-configs.md](runs-and-configs.md) says what each means. `group` names the family of the run on the Trackio dashboard.

Never give a name to other settings. `preview` then reports the old run as another model and asks for `--replace`.

## Upload a Run by Hand

`run` uploads a trained run on its own. To upload a run folder directly, for example one that is not in the matrix:

```sh
uv run confhead train upload /scratch1/$USER/confhead-runs/<run> --repo csci566-depthconf/confhead
```

The command refuses a run that has not trained every epoch, unless you pass `--partial`. It refuses to overwrite a run with another fingerprint, unless you pass `--replace`. With no run folder it rewrites the model card alone, and with `--print-card` it prints the card and uploads nothing.

## If Something Goes Wrong

| Message | What to do |
|---|---|
| `the login ... cannot write to ...` | The message gives your role in the organization and the role of the token. Ask an admin of `csci566-depthconf` for the `write` role, or log in with a write token. |
| `... holds ... entries, and run-matrix.toml expects 204894` | `data.cache` points to a cache that is not complete, or to another copy. |
| `tracked files have changes` | Commit or undo the changes, then send the command again. |
| `<run>: ... holds another model; --archive-stale renames the folder` | The run folder on disk was trained with other settings or another cache. Pass `--archive-stale`. |
| `<run>: the repository holds another model under this name` | Pass `--replace` to overwrite it, or give the run another name in the manifest. |
| `... is a run with other settings: ...` | A job found a run folder that changed after `run` checked it. The message names what differs. |
| `unknown setting ...` | A `set` of the manifest is spelled wrong. The message lists the settings of that section. |
| `torch ... has no kernels for <GPU>` | The installed torch does not support this GPU. Run `git pull` and `uv sync` on the login node, then send `run` again. |
| `upload of ... failed: ...` | An upload after an epoch failed. The training goes on, and the next epoch uploads again. |
| `<run> is trained but not in ...` | The last upload failed three times. Send `run` again: it uploads the run from the login node. |
| `... the class weight needs both classes` | The sample of train images holds no label above `tau`, or none below it. Check `inputs.labels.tau`. |

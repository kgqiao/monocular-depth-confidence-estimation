# Build the cache on CARC

This guide builds the cache for the full nuScenes v1.0-trainval on CARC. That is 850 scenes and 204,894 camera images. One command submits the build. You can run it again to finish any part that did not complete.

## What you need

- The nuScenes v1.0-trainval in `/scratch1/$USER/nuscenes`. [`utilities/dataset/download_nuscenes.sbatch`](../../utilities/dataset/download_nuscenes.sbatch) puts it there.
- The repository cloned on CARC. This guide uses `~/Vision-Project-CSCI-566`.
- [uv](https://docs.astral.sh/uv/) on CARC. To install it:

  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- About 1.1 TB of free space on `/scratch1`.

Run every command on a CARC login node, from the `confhead` folder:

```sh
cd ~/Vision-Project-CSCI-566/confhead
git pull
```

## 1. Set your project account

Slurm charges each job to a CARC project account. To see your accounts:

```sh
myaccount
```

Set the account once, in `~/.bashrc`, so every job uses it:

```sh
echo 'export SBATCH_ACCOUNT=<project>' >> ~/.bashrc
source ~/.bashrc
```

> [!NOTE]
> Replace `<project>` with an account from `myaccount`. If `SBATCH_ACCOUNT` is not set, the submit script stops and tells you.

## 2. Download the model checkpoint

```sh
mkdir -p /scratch1/$USER/checkpoints
curl -L -o /scratch1/$USER/checkpoints/depth_anything_v2_vitb.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
```

## 3. Look at the config

[`configs/nuscenes-trainval.toml`](../configs/nuscenes-trainval.toml) holds every setting of the build. These are the settings you are most likely to change:

| Setting | Value | What it is |
|---|---|---|
| `source.dataroot` | `/scratch1/$USER/nuscenes` | The folder that holds `v1.0-trainval/`, `samples/` and `sweeps/` |
| `model.checkpoint` | `/scratch1/$USER/checkpoints/depth_anything_v2_vitb.pth` | The file from step 2 |
| `output.root` | `/scratch1/$USER/confhead-cache/nuscenes-trainval` | Where the cache goes |
| `grid.patch_multiple` | `4` | The size of the grid. 4 gives 264 x 148 and about 1.1 TB. 2 gives 132 x 74 and about 290 GB. |

The `$USER` in a path changes to your CARC user name, so the same file works for every member of the team.

> [!TIP]
> To change a setting for one run only, add `--set section.key=value` to a command. [`src/confhead/cache/config.py`](../src/confhead/cache/config.py) says what each setting does.

## 4. Check the plan

The plan reads the dataset and tells you what the build will do. It does not use a GPU. The trainval tables need several GB of memory, so run the plan as a short CPU job, not on the login node:

```sh
uv sync
srun --account=$SBATCH_ACCOUNT --cpus-per-task=2 --mem=32G --time=00:30:00 \
  uv run examples/build_cache.py --config configs/nuscenes-trainval.toml --plan
```

> [!NOTE]
> `srun` does not read `SBATCH_ACCOUNT`, so the command sets the account with `--account`.

The job can wait in the queue before it starts. Then the plan takes a few minutes, and it prints a line after each step:

```
reading the v1.0-trainval tables and listing the files in /scratch1/<user>/nuscenes
850 scenes on disk, 0 skipped, in ... s
204894 items in ... s
```

Then read the first lines of the plan:

```
scenes       850: scene-0001, scene-0002, scene-0003, ..., scene-1110
splits       train 700 scenes, val 150 scenes
items        204894 camera images, 0 already built
...
             ... per entry, ... in all, ... still to write
```

> [!WARNING]
> If the plan shows fewer than 850 scenes, some of the dataset is not on disk. The output names the missing scenes on the `skipped` line. Finish the download before you continue.

## 5. Do a test run on 50 images

This job builds 50 entries on one GPU. It finds problems with the checkpoint, the GPU, or the paths before you submit 16 jobs. The full build keeps these 50 entries and does not build them again.

```sh
mkdir -p logs
sbatch --time=00:30:00 examples/carc-build.sbatch configs/nuscenes-trainval.toml --limit 50
```

`sbatch` prints the job number: `Submitted batch job <job>`. When the job ends, read its log:

```sh
cat logs/confhead-build-<job>_*.out
```

The log must end with `problems     none`. The log also shows a rough speed, on the line that starts with `50/50 items`:

```
50/50 items  <speed> items/s  eta 0.0 min
```

> [!IMPORTANT]
> Use this speed to choose the number of shards. Each task has a 12 hour limit, and one task builds 204,894 / shards items. For example, 16 shards give 12,806 items to each task. At 0.5 items/s, one task takes about 7 hours. If your estimate is more than 12 hours, use more shards.

## 6. Submit the build

```sh
examples/carc-submit.sh configs/nuscenes-trainval.toml 16
```

The script submits two jobs:

- **The build job.** It has 16 GPU tasks, and each task builds one shard.
- **The check job.** It starts when all the build tasks have ended. It writes the index of the cache and checks every entry. It runs on CPU and builds nothing.

The script prints the job numbers and the log files.

## 7. Watch the jobs

```sh
squeue --me
tail -f logs/confhead-build-<job>_0.out
```

Each build task first reads the tables, as the plan does, which takes a few minutes. Then its log prints the speed and the time left after every 25 items. To stop all tasks, use `scancel <job>`.

## 8. Read the check

When the check job ends, read its log:

```sh
cat logs/confhead-check-<job>.out
```

The cache is complete when the log ends with `problems     none` and has no `missing` line.

If a build task stopped early, for example at its time limit, the log ends with a line like this:

```
missing      8400 of 204894 entries; submit the build again to finish them
```

To finish the cache, submit the build again with the same command:

```sh
examples/carc-submit.sh configs/nuscenes-trainval.toml 16
```

Each task skips the entries that are already built and continues with the rest.

## If something goes wrong

| Message | What to do |
|---|---|
| `SBATCH_ACCOUNT: set it to your CARC project account first` | Do step 1. |
| `... does not exist; the dataroot must hold the v1.0-trainval/ tables` | Make `source.dataroot` point to the folder that holds `v1.0-trainval/`. |
| `... holds a cache built with other settings` | You changed a setting that changes the content of an entry. Set a new `output.root`, or add `--overwrite` to build every entry again. |
| `... holds no cache yet; build it first` | The check job ran before any build task wrote an entry. Read the build logs for the error. |
| `torch ... has no kernels for <GPU>` | The installed torch does not support this GPU. Run `git pull` and `uv sync` on the login node, then submit again. |
| A build log ends with `failed: <entry>: ...` | The log names each entry that failed and the cause. Fix the cause, then submit again. |

## Build the test split

`v1.0-test` is a separate version of nuScenes, so it needs its own cache. Copy the config and change two settings:

```sh
cp configs/nuscenes-trainval.toml configs/nuscenes-test.toml
```

In `configs/nuscenes-test.toml`, set `version = "v1.0-test"` and set `output.root` to a new folder, for example `/scratch1/$USER/confhead-cache/nuscenes-test`. Delete the `[splits]` section. Then do steps 4 to 8 with the new config.

# Back up the cache on Hugging Face

This guide backs up a built cache to a public Hugging Face dataset repository, and restores it from there. The full trainval cache at `grid.patch_multiple = 2` is about 340 GB. In the repository it is about 296 GB.

## How the backup works

`confhead cache upload` cuts the cache into shards of 25 scenes, in scene-name order. Each shard is a tar file compressed with zstd, about 10 GB for trainval. The command writes one shard to disk, uploads it, and deletes it. Then it writes the next one. So the upload needs only about 10 GB of free disk.

The repository holds one folder for each cache, named after the cache folder:

```
<repo>/
  README.md                            the dataset card
  nuscenes-trainval/
    manifest.json                      the shards, with the scenes, the size, and the SHA-256 of each
    cache.json, config.toml, index.json
    shards/00000.tar.zst ... 00033.tar.zst
```

Each shard goes up in one commit with the manifest. If the upload stops, submit it again: it skips each shard that is already in the repository with the same files. Run it again after a rebuild, and it sends only the shards with changed scenes.

> [!NOTE]
> zstd makes the cache only about 13% smaller, because most of the cache is float16 decoder features. A higher zstd level does not help. Measure your cache with `confhead cache info` before you upload it.

## What you need

- A built cache with a current index. `confhead cache info` shows `current` on its `index` line.
- A Hugging Face account, and an access token with write permission. To make a token, go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- The repository cloned on CARC, with `uv sync` done. `uv sync` installs the `hf` command, so you do not need a system install.

Run every command from the `confhead` folder:

```sh
cd ~/Vision-Project-CSCI-566/confhead
git pull
uv sync
```

## 1. Set the repository

In [`configs/huggingface.toml`](../configs/huggingface.toml), set `repo_id` to the repository that you want. The upload makes the repository if it does not exist.

```toml
[huggingface]
repo_id = "nadayusc/confhead-cache"
```

> [!IMPORTANT]
> The token does not go in this file, because the file is in git. Step 2 keeps the token in your home folder.

## 2. Log in

Log in once on the login node, and paste your token when the command asks for it:

```sh
uv run hf auth login
uv run hf auth whoami
```

`whoami` must show your user name. The token is now in `~/.cache/huggingface/token`, where every job can read it.

## 3. Look at the cache

```sh
uv run confhead cache info /scratch1/$USER/confhead-cache/nuscenes-trainval
```

The output shows the size and the entry count:

```
cache        /scratch1/<user>/confhead-cache/nuscenes-trainval  (fingerprint ..., created ...)
source       v1.0-trainval, model vitb tap output_conv1, grid.patch_multiple 2, fp32
size         ... GB in all, ... GB of entries
on disk      204894 complete entries in 850 scenes (204894 .npz, 204894 .json, 0 unfinished temp files)
index        204894 entries (train: ..., val: ...), 0 stale left out, current
```

If the `index` line does not show `current`, write the index first:

```sh
sbatch examples/carc-check.sbatch configs/nuscenes-trainval.toml
```

## 4. Submit the upload

The upload takes several hours, so run it as a job and not on the login node:

```sh
mkdir -p logs
sbatch examples/carc-upload.sbatch /scratch1/$USER/confhead-cache/nuscenes-trainval
```

The log prints one line for each shard:

```
nuscenes-trainval: 34 shards of 25 scenes, none in nadayusc/confhead-cache yet
shard 00000 (1/34): ... files, ... GB to ... GB (87%), packed in ... s, sent in ... s
```

When the job ends, the log shows a summary:

```
repository   https://huggingface.co/datasets/nadayusc/confhead-cache/tree/main/nuscenes-trainval
shards       34: 34 uploaded, 0 unchanged
size         ... GB of cache files in ... GB of shards (87%)
time         ... min
```

If the job stopped before the summary, submit it again with the same command.

> [!WARNING]
> Do not build the cache and upload it at the same time. Files that change during the upload can make a shard that does not match the manifest.

## 5. Restore a cache

This command downloads each shard, checks its SHA-256 against the manifest, extracts it, and deletes it:

```sh
uv run confhead cache download nuscenes-trainval /scratch1/$USER/confhead-cache/nuscenes-trainval
```

The second argument is the folder to restore into. Without it, the cache goes into `outputs/cache/<name>`. If the download stops, run it again: it skips each shard whose files are already on disk.

Without confhead, `hf` and `tar` restore the cache too:

```sh
hf download nadayusc/confhead-cache --repo-type dataset --include "nuscenes-trainval/*" --local-dir .
for f in nuscenes-trainval/shards/*.tar.zst; do tar --zstd -xf "$f" -C nuscenes-trainval; done
rm -r nuscenes-trainval/shards
```

This needs twice the disk, because all the shards are on disk before the extraction.

## Storage limits and terms

- Hugging Face gives free public storage on a best-effort basis, with no fixed limit. It asks that large datasets be useful to other people, and have a dataset card. The upload writes a short card when the repository has none. See [Storage limits](https://huggingface.co/docs/hub/storage-limits).
- A free account holds only 100 GB of private storage, so the trainval cache must go into a public repository.
- The cache derives from nuScenes, so the nuScenes license (CC BY-NC-SA 4.0) and the nuScenes terms of use apply to it. The dataset card says this.

## If something goes wrong

| Message | What to do |
|---|---|
| `Set huggingface.repo_id in the backup config` | Do step 1. |
| `index.json is missing or out of date` | Write the index, as in step 3. |
| `401 Client Error` or `Invalid user token` | Do step 2 again, with a token that has write permission. |
| `SHA-256 differs from the manifest` | The download is damaged. Run the download again. |
| `incomplete   the upload of this cache did not finish` | The download restored only the shards that are up. Finish the upload, then run the download again. |

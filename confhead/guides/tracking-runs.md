# Track the Runs in Trackio

This guide draws the loss curves of every training run in a [Trackio](https://github.com/gradio-app/trackio) dashboard on your own computer. The training jobs on CARC upload each run to the Hugging Face model repository [`nadayusc/confhead`](https://huggingface.co/nadayusc/confhead), as [training-on-carc.md](training-on-carc.md) describes. That repository is the only record of a run. The Trackio database is rebuilt from it at every sync, so you can delete the database at any time and lose nothing.

## How the Sync Works

[`examples/sync_trackio.py`](../examples/sync_trackio.py) downloads the `config.json`, `history.json`, and `steps.json` of each run folder in the repository. Hugging Face caches the files, so a sync downloads only what changed. The script then deletes the Trackio project `confhead` and logs every run into it again:

| Trackio | From |
|---|---|
| run name | the folder of the run in the repository, which is its `output.name` |
| group | `output.group`; empty for runs from before the setting |
| config | every setting of `config.json` as flat keys, such as `train.lr` and `inputs.radar`, with the best epoch and its dev loss |
| `train/loss` | `steps.json`: the mean train loss five times per epoch; absent for runs from before the file |
| `train/epoch_loss`, `dev/loss`, `lr`, `minutes`, `epoch` | `history.json`, at the last step of each epoch |

The x axis is the optimizer step. Trackio stamps each point with the time of the sync, so the time axis means nothing; the `minutes` metric holds the time of each epoch.

## 1. Sync and Open the Dashboard

From the `confhead` folder:

```sh
uv run --group tracking examples/sync_trackio.py
```

The script prints one line per run and opens the dashboard in the browser. In the dashboard, group the runs by `group` or by any setting, such as `inputs.radar`, to compare an ablation with its baseline.

Run the same command again to pick up new epochs and new runs. To rebuild the database without opening the dashboard, add `--no-show`.

## 2. Open the Dashboard Without a Sync

The database is in `outputs/trackio`. To open it as it is:

```sh
TRACKIO_DIR=outputs/trackio uv run --group tracking trackio show --project confhead
```

## Notes

- Trackio is in the `tracking` dependency group of `pyproject.toml`, pinned to one version, so the CARC jobs do not install it.
- Set `TRACKIO_DIR` to keep the database somewhere other than `outputs/trackio`.
- `--repo` syncs from another model repository, and `--project` names another Trackio project.

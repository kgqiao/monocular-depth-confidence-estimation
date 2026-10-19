# confcache

## About

The confidence head trains on a few features downstream of the Depth Anything prediction:
- DA2 decoder features
- DA2 prediction (depth)
- Temporal depth spread over the neighbor frames
- lidar and radar points (labels)

Our future changes/experiemnts to the confidence head architecture does not require changes to these input features, so the input features are calculated once and cached. This allows training and eval to run much faster without a full DA2 forward pass each time.

The `confcache` tool builds this cache from nuScenes.  For every camera image of every keyframe, it keeps the inputs of the head from [Form B](../../Prepreposal/Brainstorm/Project%20Scope%20Visual%20(Form%20B).md)

#TODO question - what from pipeline is used?

It uses the nuScenes loader and the vendored model of [`pipeline/`](../../pipeline/README.md) (`confhead`).

![One cache entry: the image with its lidar, the predicted depth, the temporal spread, the features, the lidar labels, and the radar residual](docs/preview.png)

Each block is one grid cell (here 132 x 74, the local build at `grid.patch_multiple = 2`). The pedestrians move, so their temporal spread is high, and the lidar labels show where the depth is wrong by more than 10%.

## Quickstart

### Prerequisites

- [uv](https://docs.astral.sh/uv/). (Python 3.12.)
- nuScenes v1.0-mini with its `sweeps/` folder. The config expects it at `datasets/nuscenes/v1.0-mini` in the repository.
- The Depth Anything V2 ViT-B checkpoint, [`depth_anything_v2_vitb.pth`](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth).

### Setup

```sh
cd train-v1/cache-pipeline
uv sync
export DEPTH_ANYTHING_CHECKPOINT=/path/to/depth_anything_v2_vitb.pth
```

### Usage

See what a build will do and how much disk it needs, before you run the model:

```sh
uv run confcache plan
```

```
items        2424 camera images, 0 already built
model        vitb, input 924 x 518, 5 passes per item
grid         264 x 148, tap output_conv1 (64 channels)
size         6.3 MB per entry, 15.33 GB in all, 15.33 GB still to write
```

Build the cache. Try a few items first with `--limit`:

```sh
uv run confcache build --limit 12
uv run confcache build
```

Check every entry and print statistics. `--summary` also writes them to a markdown file for the team:

```sh
uv run confcache verify --summary results/nuscenes-mini.md
```

Draw entries as figures like the one above (image and lidar, predicted depth, temporal spread, features, lidar labels, radar residual). Without keys, it draws a spread of `--count` entries into `<root>/previews`:

```sh
uv run confcache preview scene-0553_020_CAM_FRONT
uv run confcache preview --count 12
```

Run the tests. The tests on real data run when the dataset is at its default place, and the build test runs when `DEPTH_ANYTHING_CHECKPOINT` is set:

```sh
uv run pytest
```

### Settings

See [`configs/nuscenes-mini.toml`](configs/nuscenes-mini.toml) and [`config.py`](src/confcache/config.py). 

**Changing settings**

```sh
uv run confcache build --set grid.patch_multiple=2 --set output.root=outputs/cache/mini-g2
uv run confcache config --set model.tap=refinenet1
```

**Fingerprinting**

A fingerprint identifies settings that change the content of an entry. A build into a cache with another fingerprint stops and names the settings that differ. 

**On fingerprint conflicts**

Use a different `output.root`, or `--overwrite` to build every entry again. 

**Highlighted Settings**

| Setting | Default | Effect |
|---|---|---|
| `model.tap` | `output_conv1` | The decoder layer to keep: `output_conv1` (64 channels), `refinenet1` or `refinenet2` (128) |
| `grid.patch_multiple` | `4` | Grid of the dense maps: the 37 x 66 patch grid times this. 4 gives 148 x 264. It sets most of the disk size. |
| `temporal.offsets` | `[-2, -1, 1, 2]` | The 12 Hz camera frames around the keyframe |
| `temporal.align` | `lidar` | The points that give the keyframe its metric scale, for moving pixels between frames |
| `points.radar_sweeps` | `5` | Radar sweeps per sensor, the keyframe sweep and the ones before it |
| `run.precision` | `fp32` | `fp16` or `bf16` is faster on CUDA, and changes the fingerprint |

### On CARC

[`scripts/carc-build.sbatch`](scripts/carc-build.sbatch) runs the build as a job array. Each task builds one shard (`--shard i/n`), so the shards run on different GPUs at the same time:

```sh
mkdir -p logs
sbatch --account=<project> --array=0-7 scripts/carc-build.sbatch
```

Each task writes the index when it ends. Run `uv run confcache index` once all the tasks end, to be sure the index holds every entry.

## Confcache API Usage

Use the `confcache` API to read data from the cache.

```python
from confcache.store import Cache
from confcache.dataset import ConfidenceDataset, HeadInputs

cache = Cache("outputs/cache/nuscenes-mini")
train = ConfidenceDataset(cache, tag="train")
sample = train[0]
sample["x"]       # (64 + 4, 148, 264): features, depth, temporal spread, radar residual, radar validity
sample["target"]  # (1, 148, 264): 1 where the AbsRel against lidar is above tau (0.10)
sample["mask"]    # (1, 148, 264): 1 where a lidar label exists

ablation = ConfidenceDataset(cache, HeadInputs(radar=False), tag="train")
```

`Cache.rows()` and `Cache.entries()` filter by `tag`, `scene`, `channel`, `condition` (day, night), and `weather` (clear, rain). Each entry is one `.npz` of arrays and one `.json` that says what they mean. [`store.py`](src/confcache/store.py) lists the arrays.

---

## Design Policy

[design.md](design.md)

## Layout

- `src/confcache/sources/`: datasets as lists of items. `nuscenes.py` is built; `stf.py` is a stub that says what it needs.
- `src/confcache/model.py`: the frozen model and the feature tap.
- `src/confcache/build.py`: load, run, derive, and write, with resume and shards.
- `src/confcache/temporal.py`: the temporal spread.
- `src/confcache/store.py`: the layout on disk, the index, and the `Cache` reader.
- `src/confcache/derive.py`, `src/confcache/dataset.py`: labels, radar channels, and the training `Dataset`.
- `src/confcache/report.py`: `plan`, `verify`, and `preview`.
- `docs/`: the figure of this README.

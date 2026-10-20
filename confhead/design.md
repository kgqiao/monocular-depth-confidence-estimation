# Design

## The library

- **One library, scripts outside it.** `src/confhead/` holds only library code: functions and classes that return values or raise exceptions. Scripts that print, parse arguments, or save files live in `examples/`. So a notebook, a test, and a training job use the same API as the examples.
- **The `confhead` command is the one exception.** A console command must be in the package, so [`cli.py`](src/confhead/cli.py) parses the arguments and prints. Each action is one library call and its formatter, as in the examples. The command holds the tasks that work on a cache folder you already have: look at it, back it up, restore it.
- **No code before it is needed.** A setting, a dataset, or an option goes in when a task uses it. The code is the record of what we do, not of what we might do.
- **Figures are returned, not saved.** `confhead.plot` returns matplotlib figures. The caller decides where they go.
- **No nuscenes-devkit.** The loader reads the JSON tables and the point files with numpy. The radar filter is the devkit's default: `invalid_state` 0, `ambig_state` 3, `dyn_prop` 0 to 6.

## The cache

- **The cache holds what costs a model pass, or what needs the raw dataset.** The labels, the radar residual, and the radar validity are computed when an entry loads ([`labels.py`](src/confhead/labels.py), [`dataset.py`](src/confhead/dataset.py)). So you can change `tau`, the alignment, and the radar radius without a rebuild.
- **Points are kept as lists, not maps.** Each lidar and radar point is a row of (u, v, z, dt, sweep, disparity) in the pixels of the original image. A reader can put them on any grid, and can keep only the newest radar sweeps.
- **Depth is relative until a fit makes it metric.** The model gives disparity up to a scale and a shift. Each entry keeps two least-squares fits to inverse depth: one to the keyframe lidar and one to the radar. The labels use the lidar fit, as in the standard evaluation of relative depth. The radar residual uses the radar fit, because a deployed system has radar but no lidar. A fit with a scale that is not positive is refused. Then the entry has no labels: in mini, 4 night images of scene-1077, where the disparity does not correlate with the lidar at all.
- **The temporal spread corrects for ego motion.** Each grid cell is lifted to 3D with the keyframe depth and moved into each neighbor camera. The neighbor's depth there is moved back into the keyframe, and the spread is std / mean of the inverse depths. A static scene gives 0, whatever the motion. Moving objects, occlusions, and unstable depth give more. Occlusions are not removed: around thin poles in the side cameras, the spread is high from parallax alone.
- **A fingerprint guards the cache.** The settings that change what an entry holds make a fingerprint. An entry with the current fingerprint is never built again, and a build into a cache with other settings stops. Each entry's `.json` is written after its `.npz`, so it marks a complete entry.

## The backup

- **Shards of whole scenes, in name order.** 204,894 entries are 410,000 files, and a Hugging Face repository works best below 100,000. A shard of 25 scenes is about 10 GB for trainval. A scene keeps its shard when other scenes change, so a later upload sends only the shards with changed scenes.
- **Plain tar with zstd.** Any tar with zstd extracts a shard, without confhead. zstd makes the cache only about 13% smaller, because the float16 features barely compress. If the arrays were byte-shuffled first, the cache would be about 22% smaller. But then only confhead could restore it, so we use the plain format.
- **A content key decides what to upload.** The key is a hash of the path and size of each file in a shard. It needs no read of the files, so a check of 850 scenes takes seconds. Each shard goes up in one commit with the manifest, so the manifest in the repository always describes its shards, also after an upload that stopped.
- **The token stays out of the repository.** `configs/huggingface.toml` names the Hugging Face repository only. The token comes from `hf auth login` or `$HF_TOKEN`.

## Frames and units

- Depth is z along the camera's optical axis in meters, for lidar, radar, and the model, so all three compare directly. Every point is in 1 m to 80 m.
- Every 4x4 matrix maps a column vector from the frame named first to the frame named second, e.g. `sensor_to_cam`. A point moves from its sensor to the camera through the global frame, each with the ego pose at its own timestamp.
- Frames follow nuScenes: ego x forward, y left, z up; camera x right, y down, z forward. Quaternions are [w, x, y, z].
- Integer pixel coordinates are pixel centers. Where several points land on one pixel or grid cell, the nearest one wins.

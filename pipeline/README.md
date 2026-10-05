# confhead

## About

Depth Anything V2 gives a depth for every pixel of a camera image, but it does not say which pixels to trust. This project trains a small head that predicts, per pixel, when that depth is wrong. Lidar supplies the answer key, and radar is one of the inputs. See [Form B](../Prepreposal/Brainstorm/Project%20Scope%20Visual%20(Form%20B).md) and the [pre-proposal](../Prepreposal/Deliberable/CS566%20-%20Project%20Pre-proposal.pdf).

This folder is the code. It starts on nuScenes v1.0-mini. Today it reads the dataset, projects the lidar and radar returns into the six camera images, and runs Depth Anything V2. Those projections are the labels and the radar inputs of the head.

The loader, the depth predictor, and the vendored Depth Anything V2 come from [autonomous-vehicle-sandbox](https://github.com/nathanaday/autonomous-vehicle-sandbox).

## Quickstart

### Prerequisites

- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for you.
- nuScenes v1.0-mini (about 4 GB), unpacked so that one folder holds `v1.0-mini/`, `samples/`, and `sweeps/`. Download it from [nuscenes.org](https://www.nuscenes.org/nuscenes#download).
- For the model only: the Depth Anything V2 ViT-B checkpoint, [`depth_anything_v2_vitb.pth`](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth) (372 MB).

### Setup

```sh
cd pipeline
uv sync
export NUSCENES_DATAROOT=/path/to/nuscenes
export DEPTH_ANYTHING_CHECKPOINT=/path/to/depth_anything_v2_vitb.pth
```

Each command also takes `--dataroot`, and `depth-check` takes `--checkpoint`, in place of the variables.

### Usage

List the scenes, with their keyframe counts and day or night condition:

```sh
uv run confhead scenes
```

Draw the lidar (colored near to far) and the radar (magenta circles, labeled with depth) on the first keyframe of each scene, six cameras each. The images go to `outputs/overlays/`:

```sh
uv run confhead overlays
```

Measure how many pixels get a lidar or a radar return in every keyframe camera image. The per-image table goes to `outputs/projection_density.csv`, and the summary goes to [`results/projection-density.md`](results/projection-density.md):

```sh
uv run confhead report
```

Run Depth Anything V2 once on one front camera image, and print the output size, the device, and the time:

```sh
uv run confhead depth-check
```

Run the tests. The test that reads real data runs only when `NUSCENES_DATAROOT` is set:

```sh
uv run pytest
```

`outputs/` is not in git. `results/` holds the small summaries that the team reads.

## Conventions

- Depth means z along the camera's optical axis in meters, for lidar and for radar, so the two compare directly with each other and with the model.
- Every keyframe uses the ego frame at the lidar timestamp. Frames follow nuScenes: x forward, y left, z up, and quaternions are [w, x, y, z].
- A projection keeps points from 1 m to 80 m. Where several points land on one pixel, the nearest one wins.
- No nuscenes-devkit: the loader reads the JSON tables and the point files with numpy.

## Layout

- `src/confhead/nuscenes.py`: tables, calibration, and point clouds.
- `src/confhead/project.py`: points to a sparse depth map and a mask.
- `src/confhead/overlay.py`, `src/confhead/report.py`: the overlays and the density report.
- `src/confhead/depth.py`: the Depth Anything V2 predictor. The model is in `src/confhead/depth_anything_v2/` (Apache 2.0, from [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)); `dpt.py` is modified, and its header says how.

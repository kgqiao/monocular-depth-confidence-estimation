# Cache summary

Written by `confcache verify --summary`.

```
cache        /Users/nathanaday/projects/usc/kqiao-cs566-project-repo/train-v1/cache-pipeline/outputs/cache/nuscenes-mini-g2
entries      2424 (0 stale files left out), 5.75 GB on disk
tags         train: 1938, val: 486
points       3613 lidar and 150 radar per entry
fits         lidar 99.8%, radar 86.5% of entries; mean lidar fit RMSE 0.0241 1/m
labels       26.16% of grid cells hold a lidar label; 37.2% of them have AbsRel > 0.1
AbsRel       median over entries of the median per entry: day 0.052, night 0.079
temporal     ok: 2420, no lidar fit: 4; 99.5% of cells seen by 2+ frames, median relstd 0.0208
problems     none
```

Settings:

```toml
[source]
kind = "nuscenes"
dataroot = "/Users/nathanaday/projects/usc/kqiao-cs566-project-repo/datasets/nuscenes/v1.0-mini"
version = "v1.0-mini"
cameras = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
scenes = []
max_samples_per_scene = 0

[model]
checkpoint = "/Users/nathanaday/projects/software/autonomous-vehicle-sandbox/data/models/depth-anything-v2/depth_anything_v2_vitb.pth"
encoder = "vitb"
short_side = 518
tap = "output_conv1"

[grid]
patch_multiple = 2

[temporal]
enabled = true
offsets = [-2, -1, 1, 2]
align = "lidar"
min_fit_points = 20
min_overlap = 0.3
keep_neighbors = false

[points]
min_depth = 1.0
max_depth = 80.0
lidar_sweeps = 1
radar_sweeps = 5

[output]
root = "/Users/nathanaday/projects/usc/kqiao-cs566-project-repo/train-v1/cache-pipeline/outputs/cache/nuscenes-mini-g2"
feature_dtype = "float16"
store_full_disparity = true
compress = false

[run]
device = "auto"
precision = "fp32"
batch_size = 10
num_workers = 4
writer_threads = 2
log_every = 200

[splits]
train = ["scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796", "scene-1077", "scene-1094", "scene-1100"]
val = ["scene-0103", "scene-0916"]
```

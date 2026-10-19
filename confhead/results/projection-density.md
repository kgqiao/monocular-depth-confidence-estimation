# Lidar and radar density in the camera images

Written by `examples/projection_density.py`. nuScenes v1.0-mini: 10 scenes (day 7, night 3), 404 keyframes, 2424 camera images of 1600 x 900.

Each image gets the keyframe lidar sweep and the keyframe returns of all five radars (the devkit's default filter: `invalid_state` 0, `ambig_state` 3, `dyn_prop` 0 to 6), with no earlier sweeps. Points count from 1 m to 80 m in depth along the optical axis. A pixel counts once, however many points land on it. All values are means per image.

## Per camera

| Channel | Images | Lidar pixels | Lidar fraction | Radar points | Radar pixels | Radar fraction |
|---|---:|---:|---:|---:|---:|---:|
| CAM_FRONT_LEFT | 404 | 3342 | 0.232% | 15.3 | 15.2 | 0.0011% |
| CAM_FRONT | 404 | 3017 | 0.210% | 40.7 | 40.3 | 0.0028% |
| CAM_FRONT_RIGHT | 404 | 3187 | 0.221% | 18.5 | 18.3 | 0.0013% |
| CAM_BACK_LEFT | 404 | 3998 | 0.278% | 9.1 | 9.0 | 0.0006% |
| CAM_BACK | 404 | 4521 | 0.314% | 86.3 | 85.0 | 0.0059% |
| CAM_BACK_RIGHT | 404 | 3611 | 0.251% | 15.4 | 15.2 | 0.0011% |
| all | 2424 | 3613 | 0.251% | 30.9 | 30.5 | 0.0021% |

## Per condition

| Condition | Images | Lidar pixels | Lidar fraction | Radar points | Radar pixels | Radar fraction |
|---|---:|---:|---:|---:|---:|---:|
| day | 1698 | 3582 | 0.249% | 31.7 | 31.3 | 0.0022% |
| night | 726 | 3685 | 0.256% | 29.0 | 28.7 | 0.0020% |
| all | 2424 | 3613 | 0.251% | 30.9 | 30.5 | 0.0021% |

61 of 2424 images get no radar point.

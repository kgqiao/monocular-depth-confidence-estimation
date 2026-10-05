# Lidar and radar density in the camera images

Written by `confhead report`. nuScenes v1.0-mini: 10 scenes (day 7, night 3), 404 keyframes, 2424 camera images of 1600 x 900.

Each image gets the keyframe lidar sweep and the keyframe returns of all five radars (devkit filter: `invalid_state` 0, `ambig_state` 3), with no earlier sweeps. Points count from 1 m to 80 m in depth along the optical axis. A pixel counts once, however many points land on it. All values are means per image.

## Per camera

| Channel | Images | Lidar pixels | Lidar fraction | Radar points | Radar pixels | Radar fraction |
|---|---:|---:|---:|---:|---:|---:|
| CAM_FRONT_LEFT | 404 | 3342 | 0.232% | 15.5 | 15.3 | 0.0011% |
| CAM_FRONT | 404 | 3017 | 0.210% | 41.1 | 40.7 | 0.0028% |
| CAM_FRONT_RIGHT | 404 | 3187 | 0.221% | 18.7 | 18.6 | 0.0013% |
| CAM_BACK_LEFT | 404 | 3998 | 0.278% | 9.2 | 9.1 | 0.0006% |
| CAM_BACK | 404 | 4521 | 0.314% | 87.1 | 85.8 | 0.0060% |
| CAM_BACK_RIGHT | 404 | 3611 | 0.251% | 15.6 | 15.4 | 0.0011% |
| all | 2424 | 3613 | 0.251% | 31.2 | 30.8 | 0.0021% |

## Per condition

| Condition | Images | Lidar pixels | Lidar fraction | Radar points | Radar pixels | Radar fraction |
|---|---:|---:|---:|---:|---:|---:|
| day | 1698 | 3582 | 0.249% | 32.0 | 31.6 | 0.0022% |
| night | 726 | 3685 | 0.256% | 29.3 | 28.9 | 0.0020% |
| all | 2424 | 3613 | 0.251% | 31.2 | 30.8 | 0.0021% |

59 of 2424 images get no radar point.

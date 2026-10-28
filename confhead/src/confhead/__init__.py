"""confhead: a per-pixel confidence head for the depth of a frozen Depth Anything V2.

Modules:
    nuscenes    the nuScenes tables, sensor files, and camera items
    items       one camera image with its neighbors and point files; point projection
    geometry    frames, projection, pixel grids, sampling, scale-and-shift fit
    depth       the frozen Depth Anything V2 with a decoder feature tap
    cache       build, check, read, and back up the cache of model outputs
    labels      lidar error labels
    dataset     the head inputs and the PyTorch dataset over a cache
    head        the confidence head, its masked loss, and its checkpoints
    train       the training settings, the dev split, and one training run
    plot        figures of items and cache entries
    cli         the confhead command
"""

from .cache import Cache, CacheConfig, build, load_config
from .dataset import ConfidenceDataset, HeadInputs, RadarConfig
from .depth import FrozenDepthAnything
from .head import ConfidenceHead, load_head
from .items import Item, load_points
from .labels import LabelConfig
from .nuscenes import NuScenes
from .train import TrainConfig, Training, load_train_config

__all__ = [
    "Cache", "CacheConfig", "ConfidenceDataset", "ConfidenceHead", "FrozenDepthAnything", "HeadInputs", "Item",
    "LabelConfig", "NuScenes", "RadarConfig", "TrainConfig", "Training", "build", "load_config", "load_head",
    "load_points", "load_train_config",
]

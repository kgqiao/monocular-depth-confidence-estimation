"""Frozen Depth Anything V2: the disparity and one decoder feature map per image.

The model predicts relative inverse depth: disparity up to an unknown scale
and shift, larger for nearer surfaces.

Taps, as multiples of the patch grid (input size / 14), for ViT-B:
- refinenet2: 128 channels at 4x, the second-to-last fusion block.
- refinenet1: 128 channels at 8x, the last fusion block.
- output_conv1: 64 channels at 8x, the last decoder features before the
  depth regression convolutions.
Channel counts scale with the encoder: `features` and `features // 2` of MODEL_CONFIGS.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# The vendored DINOv2 logs a warning at import when xFormers is missing; it is optional.
logging.getLogger("dinov2").setLevel(logging.ERROR)

from .depth_anything_v2.dpt import DepthAnythingV2

CHECKPOINT_ENV = "DEPTH_ANYTHING_CHECKPOINT"
PATCH = 14

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
TAP_MODULES = {
    "refinenet2": "depth_head.scratch.refinenet2",
    "refinenet1": "depth_head.scratch.refinenet1",
    "output_conv1": "depth_head.scratch.output_conv1",
}
PRECISIONS = ("fp32", "fp16", "bf16")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tap_channels(encoder: str, tap: str) -> int:
    features = MODEL_CONFIGS[encoder]["features"]
    return features // 2 if tap == "output_conv1" else features


def input_size(width: int, height: int, short_side: int = 518) -> tuple[int, int]:
    """Model input (w, h): the official resize of the short side, both sides rounded to the 14 px patch."""
    scale = short_side / min(width, height)
    return round(width * scale / PATCH) * PATCH, round(height * scale / PATCH) * PATCH


def preprocess(path: str | Path, short_side: int = 518) -> np.ndarray:
    """Image file -> (3, h, w) float32, resized bicubic and normalized as upstream."""
    img = Image.open(path).convert("RGB")
    w, h = input_size(*img.size, short_side)
    x = np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        check_cuda_kernels(device)
    return device


def check_cuda_kernels(device: torch.device) -> None:
    """Raise if this torch build has no kernels for the GPU, which otherwise fails later inside a convolution."""
    major, minor = torch.cuda.get_device_capability(device)
    for arch in torch.cuda.get_arch_list():
        kind, _, cc = arch.partition("_")
        built = (int(cc[:-1]), int(cc[-1]))
        if (kind == "sm" and built[0] == major and built[1] <= minor) or (kind == "compute" and built <= (major, minor)):
            return
    raise RuntimeError(f"torch {torch.__version__} has no kernels for {torch.cuda.get_device_name(device)} "
                       f"(compute capability {major}.{minor}); it supports {' '.join(torch.cuda.get_arch_list())}. "
                       f"Run `uv sync` to install the CUDA build from pyproject.toml, or use another GPU.")


class FrozenDepthAnything:
    """`disparity, features = FrozenDepthAnything(checkpoint)(images)`."""

    def __init__(self, checkpoint: str | Path, encoder: str = "vitb", tap: str = "output_conv1",
                 device: str = "auto", precision: str = "fp32"):
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Depth Anything checkpoint {checkpoint} does not exist; "
                                    f"pass the depth_anything_v2_{encoder}.pth file.")
        if tap not in TAP_MODULES:
            raise ValueError(f"tap {tap!r}: use one of {', '.join(TAP_MODULES)}")
        if precision not in PRECISIONS:
            raise ValueError(f"precision {precision!r}: use one of {', '.join(PRECISIONS)}")
        self.device = pick_device(device)
        self.precision = precision
        model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.channels = tap_channels(encoder, tap)
        self._tapped: torch.Tensor | None = None
        self.model.get_submodule(TAP_MODULES[tap]).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        self._tapped = output

    def _autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(self.device.type, dtype=dtype)

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, h, w) from `preprocess` -> disparity (B, h, w) and features (B, C, h', w'), float32 on the device."""
        x = x.to(self.device, non_blocking=True)
        with self._autocast():
            disparity = self.model(x)
        features, self._tapped = self._tapped, None
        return disparity.float(), features.float()

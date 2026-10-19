"""Frozen Depth Anything V2 that returns the disparity and one decoder feature map.

Taps, as multiples of the patch grid (input size / 14), for ViT-B:
- refinenet2: 128 channels at 4x, the second-to-last fusion block.
- refinenet1: 128 channels at 8x, the last fusion block.
- output_conv1: 64 channels at 8x, the last decoder features before the
  depth regression convolutions.
Channel counts scale with the encoder: `features` and `features // 2` of
confhead.depth.MODEL_CONFIGS.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from confhead.depth import IMAGENET_MEAN, IMAGENET_STD, MODEL_CONFIGS, DepthAnythingV2, input_size

TAP_MODULES = {
    "refinenet2": "depth_head.scratch.refinenet2",
    "refinenet1": "depth_head.scratch.refinenet1",
    "output_conv1": "depth_head.scratch.output_conv1",
}
TAP_MULTIPLE = {"refinenet2": 4, "refinenet1": 8, "output_conv1": 8}


def tap_channels(encoder: str, tap: str) -> int:
    features = MODEL_CONFIGS[encoder]["features"]
    return features // 2 if tap == "output_conv1" else features


def model_size(width: int, height: int, short_side: int) -> tuple[int, int]:
    """Model input (w, h): the official resize of the short side, rounded to the patch."""
    return input_size(width, height, short_side)


def preprocess(path: Path, short_side: int) -> np.ndarray:
    """Image file -> (3, h, w) float32, resized bicubic and normalized as in confhead.depth."""
    img = Image.open(path).convert("RGB")
    w, h = model_size(*img.size, short_side)
    x = np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FrozenDepthAnything:
    def __init__(self, checkpoint: str | Path, encoder: str, tap: str,
                 device: str = "auto", precision: str = "fp32"):
        if not checkpoint:
            raise SystemExit("Set model.checkpoint or $DEPTH_ANYTHING_CHECKPOINT: "
                             f"the depth_anything_v2_{encoder}.pth file.")
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(f"Depth Anything checkpoint {checkpoint} does not exist.")
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
        """(B, 3, h, w) -> disparity (B, h, w) and tapped features (B, C, h', w'), float32 on the device."""
        x = x.to(self.device, non_blocking=True)
        with self._autocast():
            disparity = self.model(x)
        features, self._tapped = self._tapped, None
        return disparity.float(), features.float()

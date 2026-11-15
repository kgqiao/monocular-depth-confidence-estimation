import torch
import numpy as np
from confhead.depth import FrozenDepthAnything

def flip_confidence(model: FrozenDepthAnything, image: torch.Tensor) -> np.ndarray:
    """
    image: (1, 3, H, W) tensor from preprocess()
    Returns confidence map (H, W) in [0, 1]:
        1 = depths agree (high confidence)
        0 = depths disagree (low confidence)
    """
    disp_orig, _ = model(image)                             # (1, H, W)
    disp_flip, _ = model(torch.flip(image, dims=[-1]))      # (1, H, W)
    disp_flip    = torch.flip(disp_flip, dims=[-1])         # un-flip back

    orig = disp_orig.squeeze().cpu().numpy()
    flip = disp_flip.squeeze().cpu().numpy()

    disagreement = np.abs(orig - flip) / (np.abs(orig) + np.abs(flip) + 1e-6)
    confidence   = 1.0 - disagreement

    return confidence.astype(np.float32)
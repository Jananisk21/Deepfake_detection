"""
Grad-CAM for the dual-branch model.

Paper basis: Section 4 of the review identifies Grad-CAM as "the dominant
post-hoc explanation method applied to deepfake detectors", but also
notes (Section 4 and Section 6, Future Directions) that "explanation
methods tailored to dual-branch, frequency-fused architectures ... are
comparatively rare", and explicitly proposes "extending Grad-CAM-style
attribution separately to the semantic and frequency branches of a
cross-attention fusion network" as a concrete future direction.

Proposed implementation decision: this module implements ONE generic
Grad-CAM class and applies it to BOTH branches independently (spatial
branch -> heatmap over the RGB face; frequency branch -> heatmap over the
FFT/DCT magnitude map), directly addressing the gap above. Grad-CAM
(Selvaraju et al., 2017) rather than Grad-CAM++ is used for simplicity
(swapping in Grad-CAM++'s second-order weighting is a drop-in extension
of `_compute_weights` below if a stronger localization is later needed).

Because the trained model is a VIDEO-level model (it consumes a sequence
of frames through a GRU), Grad-CAM here is computed per individual frame
by running that one frame as a length-1 sequence through the model (same
approach used for frame-level evaluation in evaluation/evaluate.py) --
this is a deliberate simplification that keeps the attribution
well-defined per frame while still using the actual trained model end to
end (including its temporal aggregator, degenerated to a no-op on a
single-frame input).
"""
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.target_module = target_module
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_handle = target_module.register_forward_hook(self._save_activation)
        self._bwd_handle = target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0]

    def _compute_weights(self, gradients: torch.Tensor) -> torch.Tensor:
        # classic Grad-CAM: global-average-pool the gradients over spatial dims
        return gradients.mean(dim=(2, 3), keepdim=True)

    def __call__(self, rgb_1frame: torch.Tensor, freq_1frame: torch.Tensor, target: str = "fake") -> Tuple[np.ndarray, float]:
        """
        rgb_1frame:  (1, 1, 3, H, W)  a single frame as a length-1 sequence
        freq_1frame: (1, 1, 1, H, W)
        target: "fake" backprops from the fake-class logit (positive logit);
                the model outputs a single logit where higher = more fake,
                so this is simply `logit` itself.
        Returns (heatmap in [0,1] at the model's input resolution, predicted probability).
        """
        self.model.zero_grad()
        out = self.model(rgb_1frame, freq_1frame)
        logit = out["logit"]
        prob = torch.sigmoid(logit).item()
        logit.backward()

        activations = self._activations.detach()  # (1, C, h, w)
        gradients = self._gradients.detach()        # (1, C, h, w)
        weights = self._compute_weights(gradients)    # (1, C, 1, 1)

        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        target_size = rgb_1frame.shape[-1]
        cam = cv2.resize(cam, (target_size, target_size))
        return cam, prob

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def spatial_gradcam(model, rgb_1frame: torch.Tensor, freq_1frame: torch.Tensor) -> Tuple[np.ndarray, float]:
    """Grad-CAM heatmap over the RGB face for the spatial branch's last conv layer."""
    cam_module = GradCAM(model, model.spatial_branch.get_target_layer())
    heatmap, prob = cam_module(rgb_1frame, freq_1frame)
    cam_module.remove()
    return heatmap, prob


def frequency_gradcam(model, rgb_1frame: torch.Tensor, freq_1frame: torch.Tensor) -> Tuple[np.ndarray, float]:
    """Grad-CAM heatmap over the FFT/DCT magnitude map for the frequency branch's last conv layer."""
    cam_module = GradCAM(model, model.frequency_branch.get_target_layer())
    heatmap, prob = cam_module(rgb_1frame, freq_1frame)
    cam_module.remove()
    return heatmap, prob


def overlay_heatmap(image_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlays a [0,1] heatmap onto an RGB uint8 image using the 'jet' colormap."""
    if image_rgb_uint8.dtype != np.uint8:
        if image_rgb_uint8.max() <= 1.0:
            image_rgb_uint8 = np.uint8(np.clip(image_rgb_uint8 * 255.0, 0, 255))
        else:
            image_rgb_uint8 = np.uint8(np.clip(image_rgb_uint8, 0, 255))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

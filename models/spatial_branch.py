"""
Spatial (RGB) branch.

Paper basis: Section 2.1 identifies Xception as the dominant spatial-domain
backbone in the reviewed literature (and the semantic branch used inside
D4-Net), because its depthwise-separable convolutions transfer well to
capturing facial geometry, blending seams and texture inconsistencies.

Proposed implementation decision: full Xception (~22M params) is heavier
than necessary for a project that has to train on a normal student laptop
CPU. We use MobileNetV3-Small (~2.5M params, torchvision pretrained on
ImageNet) as a practical, lightweight stand-in that plays the same
architectural role (a pretrained CNN backbone producing a spatial feature
vector). EfficientNet-B0 is also wired in as a swappable option for a
stronger machine. This substitution is our engineering choice for
feasibility, not something the paper prescribes -- the paper itself
proposes no concrete architecture (it is a review).
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class SpatialBranch(nn.Module):
    """RGB branch: pretrained CNN backbone -> pooled feature vector.

    Output: (B, feature_dim) spatial feature vector per frame.
    Also exposes the last convolutional feature map (via a forward hook)
    for Grad-CAM in explainability/gradcam.py.
    """

    def __init__(self, backbone: str = "mobilenet_v3_small", pretrained: bool = True, freeze_early: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self._last_conv_features = None  # populated by the hook for Grad-CAM

        if backbone == "mobilenet_v3_small":
            weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = tvm.mobilenet_v3_small(weights=weights)
            self.features = net.features  # conv feature extractor
            self.feature_dim = 576
        elif backbone == "mobilenet_v3_large":
            weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            net = tvm.mobilenet_v3_large(weights=weights)
            self.features = net.features
            self.feature_dim = 960
        elif backbone == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = tvm.efficientnet_b0(weights=weights)
            self.features = net.features
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported spatial backbone: {backbone}")

        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_early:
            self._freeze_early_layers()

        # NOTE: we do NOT store the target conv layer as a new `self.x = ...`
        # attribute, because assigning an nn.Module to an attribute makes
        # PyTorch register it a SECOND time under a new state_dict key
        # (duplicating the same parameters under two names and breaking
        # checkpoint compatibility). get_target_layer() below just returns
        # the existing reference instead.
        self.features[-1].register_forward_hook(self._save_activation)

    def _freeze_early_layers(self, frac: float = 0.6):
        """Freezes the first `frac` of conv blocks so training on a small
        dataset (student project scale) needs fewer gradients/less compute,
        while still fine-tuning the later, more task-specific layers."""
        n = len(self.features)
        n_freeze = int(n * frac)
        for i, layer in enumerate(self.features):
            if i < n_freeze:
                for p in layer.parameters():
                    p.requires_grad = False

    def _save_activation(self, module, inp, out):
        self._last_conv_features = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) normalized RGB. Returns (B, feature_dim)."""
        feat_map = self.features(x)
        self._last_conv_features = feat_map
        pooled = self.pool(feat_map).flatten(1)
        return pooled

    def get_last_conv_features(self) -> torch.Tensor:
        """Used by Grad-CAM: the (B, C, H, W) activation map before pooling."""
        return self._last_conv_features

    def get_target_layer(self) -> nn.Module:
        """The conv module Grad-CAM should hook (see explainability/gradcam.py)."""
        return self.features[-1]

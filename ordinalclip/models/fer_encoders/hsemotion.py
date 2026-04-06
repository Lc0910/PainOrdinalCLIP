"""HSEmotion / EmotiEffNet encoder for facial expression recognition.

Uses EfficientNet-B0/B2 pretrained on VGGFace2 (face recognition) then
fine-tuned on FER datasets (AffectNet, AFEW, VGAF — depends on model variant).

Based on: Savchenko, A.V. (2022). "HSEmotion: High-Speed Emotion Recognition
Library". ICPR Workshop.

Weight sources:
    - pip install hsemotion  (auto-downloads checkpoints)
    - Manual: github.com/HSE-asavchenko/face-emotion-recognition

We bypass the HSEmotion high-level API and directly use timm EfficientNet
so that the encoder accepts standard PyTorch tensors [B, 3, H, W] and
integrates seamlessly with the training pipeline.

Interface:
    model = hsemotion_b0(num_classes=512)
    out = model(x)  # x: [B, 3, 224, 224] -> out: [B, 512]

Requires: pip install timm hsemotion
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["HSEmotionNet", "hsemotion_b0", "hsemotion_b2"]


class HSEmotionNet(nn.Module):
    """HSEmotion EfficientNet wrapper for FER feature extraction.

    Loads a timm EfficientNet model and optionally replaces weights with
    HSEmotion's VGGFace2→FER pretrained checkpoint. The classification
    head is replaced with a projection to the desired output dimension.

    Args:
        variant: 'b0' or 'b2'.
        num_classes: Output feature dimension.
        hsemotion_model_name: HSEmotion model identifier for weight loading.
            If None, uses timm's ImageNet pretrained weights.
            Examples: 'enet_b0_8_best_afew', 'enet_b0_8_best_vgaf', 'enet_b2_8'
        pretrained_path: Direct path to a .pt checkpoint. Overrides
            hsemotion_model_name if both are given.
    """

    def __init__(
        self,
        variant: str = "b0",
        num_classes: int = 512,
        hsemotion_model_name: Optional[str] = None,
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for HSEmotionNet: pip install timm")

        # EfficientNet config
        model_map = {
            "b0": ("efficientnet_b0", 1280, 224),
            "b2": ("efficientnet_b2", 1408, 260),
        }
        if variant not in model_map:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(model_map.keys())}")

        timm_name, self.backbone_dim, self.input_resolution = model_map[variant]

        # Create backbone without classification head
        self.backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        self.fer_weights_loaded = False

        # Load HSEmotion FER weights if available
        if pretrained_path is not None:
            self._load_weights_from_path(pretrained_path)
        elif hsemotion_model_name is not None:
            self._load_hsemotion_weights(hsemotion_model_name)

        # Projection to target dimension
        self.fc = nn.Linear(self.backbone_dim, num_classes, bias=False)
        self.bn = nn.LayerNorm(num_classes)  # LayerNorm: safe with any batch size

    def _load_hsemotion_weights(self, model_name: str) -> None:
        """Load weights from HSEmotion package's cached checkpoints."""
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            # HSEmotionRecognizer downloads and caches the model
            recognizer = HSEmotionRecognizer(model_name=model_name, device="cpu")
            # Extract the underlying PyTorch model's state_dict
            src_state = recognizer.model.state_dict()
            self._load_compatible_weights(src_state, source_name=f"hsemotion:{model_name}")
        except ImportError:
            raise ImportError(
                "hsemotion package is required to load FER pretrained weights. "
                "Install with: pip install hsemotion\n"
                "If you intentionally want ImageNet-only weights, set "
                "hsemotion_model_name=None in the config."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HSEmotion weights ({model_name}): {e}\n"
                "Check that the model name is valid and network is available."
            ) from e

    def _load_weights_from_path(self, path: str) -> None:
        """Load weights from a local checkpoint file."""
        logger.info(f"Loading HSEmotion weights from {path}")
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self._load_compatible_weights(state_dict, source_name=path)

    def _load_compatible_weights(self, src_state: dict, source_name: str) -> None:
        """Load compatible keys from a source state dict into the backbone."""
        model_dict = self.backbone.state_dict()
        compatible = {}
        skipped = []

        for k, v in src_state.items():
            clean_k = k.replace("module.", "")
            # Skip classification head keys (we replace it)
            if "classifier" in clean_k or "fc" in clean_k or "head" in clean_k:
                skipped.append(clean_k)
                continue
            if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
                compatible[clean_k] = v
            else:
                skipped.append(clean_k)

        # Validate: at least 50% of backbone keys must match
        min_required = len(model_dict) // 2
        if len(compatible) < min_required:
            raise RuntimeError(
                f"HSEmotion checkpoint key mismatch: only {len(compatible)}/{len(model_dict)} "
                f"backbone keys matched from {source_name} (min required: {min_required}). "
                f"This likely means the checkpoint format is incompatible with the "
                f"timm EfficientNet backbone. Sample src keys: "
                f"{list(src_state.keys())[:5]}"
            )

        model_dict.update(compatible)
        self.backbone.load_state_dict(model_dict)
        self.fer_weights_loaded = True
        logger.info(
            f"Loaded {len(compatible)}/{len(src_state)} keys from {source_name} "
            f"(skipped {len(skipped)}: head/mismatched)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> features: [B, num_classes]"""
        feat = self.backbone(x)  # [B, backbone_dim]
        out = self.fc(feat)  # [B, num_classes]
        out = self.bn(out)  # [B, num_classes]
        return out


def hsemotion_b0(
    num_classes: int = 512,
    pretrained_path: Optional[str] = None,
    hsemotion_model_name: str = "enet_b0_8_best_afew",
    **kwargs: Any,
) -> HSEmotionNet:
    """Create HSEmotion EfficientNet-B0 encoder.

    Default: VGGFace2 → AFEW pretrained (enet_b0_8_best_afew, via hsemotion package).
    Input: [B, 3, 224, 224]. Output: [B, num_classes].
    Backbone feature dim: 1280.

    Args:
        num_classes: Output feature dimension.
        pretrained_path: Path to local checkpoint (overrides hsemotion_model_name).
        hsemotion_model_name: HSEmotion model name for auto-download.
    """
    return HSEmotionNet(
        variant="b0",
        num_classes=num_classes,
        hsemotion_model_name=hsemotion_model_name,
        pretrained_path=pretrained_path,
        **kwargs,
    )


def hsemotion_b2(
    num_classes: int = 512,
    pretrained_path: Optional[str] = None,
    hsemotion_model_name: str = "enet_b2_8",
    **kwargs: Any,
) -> HSEmotionNet:
    """Create HSEmotion EfficientNet-B2 encoder.

    Default: VGGFace2 → AffectNet pretrained (enet_b2_8, via hsemotion package).
    Input: [B, 3, 260, 260]. Output: [B, num_classes].
    Backbone feature dim: 1408.

    Note: Input resolution is 260x260, not 224x224. Configure data transforms
    accordingly or use B0 variant for 224x224 compatibility.

    Args:
        num_classes: Output feature dimension.
        pretrained_path: Path to local checkpoint (overrides hsemotion_model_name).
        hsemotion_model_name: HSEmotion model name for auto-download.
    """
    return HSEmotionNet(
        variant="b2",
        num_classes=num_classes,
        hsemotion_model_name=hsemotion_model_name,
        pretrained_path=pretrained_path,
        **kwargs,
    )

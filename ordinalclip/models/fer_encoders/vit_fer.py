"""ViT-based encoder for facial expression recognition via timm.

Supports loading ViT models with FER-specific pretrained weights, e.g.
from HSEmotion (savchenko/hsemotion) or custom AffectNet fine-tuned
checkpoints.

Interface:
    model = vit_fer_base(num_classes=512)
    out = model(x)  # x: [B, 3, 224, 224] -> out: [B, 512]

Requires: pip install timm
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["ViTFER", "vit_fer_base"]


class ViTFER(nn.Module):
    """ViT wrapper for FER feature extraction using timm.

    Creates a timm ViT model, removes its classification head, and adds
    a projection layer to map to the desired output dimension.

    Args:
        model_name: timm model identifier (e.g. 'vit_base_patch16_224').
        num_classes: Output feature dimension.
        pretrained_timm: Use timm's default pretrained weights (ImageNet).
    """

    input_resolution: int = 224

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 512,
        pretrained_timm: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for ViTFER: pip install timm")

        # Create model without classification head
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained_timm, num_classes=0
        )
        self.backbone_dim = self.backbone.num_features  # e.g. 768 for ViT-B

        # Projection to target dimension
        self.fc = nn.Linear(self.backbone_dim, num_classes, bias=False)
        self.bn = nn.LayerNorm(num_classes)  # LayerNorm: safe with any batch size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 224, 224] -> features: [B, num_classes]"""
        feat = self.backbone(x)  # [B, backbone_dim]
        out = self.fc(feat)  # [B, num_classes]
        out = self.bn(out)  # [B, num_classes]
        return out


def vit_fer_base(
    num_classes: int = 512,
    pretrained_path: Optional[str] = None,
    model_name: str = "vit_base_patch16_224",
    **kwargs: Any,
) -> ViTFER:
    """Create a ViT-FER model with optional pretrained FER weights.

    Args:
        num_classes: Output feature dimension.
        pretrained_path: Path to FER fine-tuned checkpoint. If None, uses
            timm's ImageNet pretrained weights.
        model_name: timm model name.

    Returns:
        ViTFER model.
    """
    model = ViTFER(model_name=model_name, num_classes=num_classes, **kwargs)
    model.fer_weights_loaded = False

    if pretrained_path is not None:
        logger.info(f"Loading ViT-FER pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu")

        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        model_dict = model.state_dict()
        compatible = {}
        skipped = []
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            if clean_k in model_dict and model_dict[clean_k].shape == v.shape:
                compatible[clean_k] = v
            else:
                skipped.append(k)

        if skipped:
            logger.info(f"Skipped {len(skipped)} keys: {skipped[:5]}...")

        min_required = len(model_dict) // 2
        if len(compatible) < min_required:
            raise RuntimeError(
                f"ViT-FER checkpoint key mismatch: only {len(compatible)}/{len(model_dict)} "
                f"model keys matched (min required: {min_required}). "
                f"Sample checkpoint keys: {list(state_dict.keys())[:5]}"
            )

        model_dict.update(compatible)
        model.load_state_dict(model_dict)
        model.fer_weights_loaded = True
        logger.info(f"Loaded {len(compatible)}/{len(state_dict)} keys from ViT-FER checkpoint")

    return model

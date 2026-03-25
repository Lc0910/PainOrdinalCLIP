"""DAN (Distract Your Attention Network) encoder for facial expression recognition.

Based on: "Distract Your Attention: Multi-head Cross Attention Network for
Facial Expression Recognition" (FG 2022).

Architecture: ResNet-18 backbone + multi-head cross-attention for spatial
feature refinement. Pretrained on MS-Celeb-1M (face recognition) then
fine-tuned on AffectNet-7 (facial expression recognition).

Weight sources:
    - AffectNet-7: github.com/yaoing/DAN → affecnet7_epoch5_acc0.6209.pth
    - AffectNet-8: github.com/yaoing/DAN → affecnet8_epoch5_acc0.6120.pth
    - RAF-DB:      github.com/yaoing/DAN → rafdb_epoch21_acc0.8980.pth

Interface:
    model = dan_resnet18(num_classes=512)
    out = model(x)  # x: [B, 3, 224, 224] -> out: [B, 512]
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

logger = logging.getLogger(__name__)

__all__ = ["DAN", "dan_resnet18"]


class CrossAttentionHead(nn.Module):
    """Single cross-attention head from DAN.

    Computes attention weights between a global query and spatial key-value
    pairs from CNN feature maps, producing an attention-refined feature vector.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.sa = SpatialAttention()
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> attended: [B, C]"""
        att = self.sa(x)  # [B, 1, H, W]
        query = (x * att).view(x.size(0), x.size(1), -1)  # [B, C, HW]
        key = self.key_conv(x).view(x.size(0), x.size(1), -1)  # [B, C, HW]
        value = self.value_conv(x).view(x.size(0), x.size(1), -1)  # [B, C, HW]

        # Attention: [B, HW, HW]
        energy = torch.bmm(query.permute(0, 2, 1), key)  # [B, HW, HW]
        attention = self.softmax(energy)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(x.shape)  # [B, C, H, W]
        out = self.gamma * out + x  # residual

        return out.mean(dim=[2, 3])  # [B, C] global average pool


class SpatialAttention(nn.Module):
    """Lightweight spatial attention (channel squeeze → conv → sigmoid)."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> attention_map: [B, 1, H, W]"""
        avg_out = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = x.max(dim=1, keepdim=True)  # [B, 1, H, W]
        feat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        return self.sigmoid(self.conv(feat))  # [B, 1, H, W]


class DAN(nn.Module):
    """DAN: Distract Your Attention Network.

    ResNet-18 backbone with multiple cross-attention heads for facial
    expression feature extraction.

    Args:
        num_classes: Output feature dimension (default 512 for backbone output).
        num_heads: Number of cross-attention heads (default 4).
        pretrained_backbone: Load ImageNet pretrained ResNet-18 weights.
    """

    input_resolution: int = 224

    def __init__(
        self,
        num_classes: int = 512,
        num_heads: int = 4,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        backbone = resnet18(pretrained=pretrained_backbone)
        self.backbone_dim = 512  # ResNet-18 last layer output dim

        # Use ResNet-18 layers up to layer4 (before avgpool and fc)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # Multi-head cross-attention
        self.num_heads = num_heads
        self.cat_heads = nn.ModuleList(
            [CrossAttentionHead(self.backbone_dim) for _ in range(num_heads)]
        )

        # Final projection: num_heads * backbone_dim -> num_classes
        self.fc = nn.Linear(self.backbone_dim * num_heads, num_classes, bias=False)
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 224, 224] -> features: [B, num_classes]"""
        feat_map = self.features(x)  # [B, 512, 7, 7]

        # Apply each cross-attention head
        head_outputs = [head(feat_map) for head in self.cat_heads]  # list of [B, 512]
        concat = torch.cat(head_outputs, dim=1)  # [B, 512 * num_heads]

        out = self.fc(concat)  # [B, num_classes]
        out = self.bn(out)  # [B, num_classes]
        return out


def dan_resnet18(
    num_classes: int = 512,
    pretrained_path: Optional[str] = None,
    **kwargs: Any,
) -> DAN:
    """Create a DAN model with optional pretrained FER weights.

    Args:
        num_classes: Output feature dimension.
        pretrained_path: Path to DAN pretrained checkpoint (.pth).
            If None, only ImageNet backbone weights are used.

    Returns:
        DAN model ready for feature extraction.
    """
    model = DAN(num_classes=num_classes, **kwargs)
    model.fer_weights_loaded = False

    if pretrained_path is not None:
        logger.info(f"Loading DAN pretrained weights from {pretrained_path}")
        # weights_only=False: FER checkpoints may contain non-tensor metadata
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)

        # DAN checkpoint may have different key names or extra keys
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict:
            state_dict = state_dict["model"]

        # Filter out mismatched keys (fc layer may differ in num_classes)
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
            logger.info(f"Skipped {len(skipped)} keys with shape mismatch: {skipped[:5]}...")

        # Validate: at least 50% of model keys must be loaded
        min_required = len(model_dict) // 2
        if len(compatible) < min_required:
            raise RuntimeError(
                f"DAN checkpoint key mismatch: only {len(compatible)}/{len(model_dict)} "
                f"model keys matched (min required: {min_required}). "
                f"Check checkpoint format. Sample checkpoint keys: "
                f"{list(state_dict.keys())[:5]}"
            )

        model_dict.update(compatible)
        model.load_state_dict(model_dict)
        model.fer_weights_loaded = True
        logger.info(f"Loaded {len(compatible)}/{len(state_dict)} keys from DAN checkpoint")

    return model

"""Siamese Ranking Head for OrdinalCLIP / Baseline backbone.

Stage 2 model: frozen backbone extracts features from image pairs,
RankingHead MLP predicts pairwise ordering P(rank_A > rank_B).

Two head variants:
  - LinearRankingHead:  D -> 1  (small-capacity baseline, P1-2)
  - MLPRankingHead:     D -> hidden -> hidden//2 -> 1  (full capacity)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ordinalclip.utils import get_logger

from .builder import MODELS

logger = get_logger(__name__)


class LinearRankingHead(nn.Module):
    """Minimal linear ranking head (D -> 1).  Serves as capacity lower-bound."""

    def __init__(self, embed_dims: int, **kwargs) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dims, 1)

    def forward(self, feat_diff: torch.Tensor) -> torch.Tensor:
        return self.linear(feat_diff)  # [B, 1]


class MLPRankingHead(nn.Module):
    """3-layer MLP ranking head.

    Input:  feat_diff = feat_A - feat_B  [B, D]
    Output: ranking_logit [B, 1]  (sigmoid -> P(A > B))
    """

    def __init__(
        self,
        embed_dims: int,
        hidden_dims: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),          # [B, D] -> [B, hidden]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims // 2),    # [B, hidden] -> [B, hidden//2]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims // 2, 1),              # [B, hidden//2] -> [B, 1]
        )

    def forward(self, feat_diff: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat_diff)  # [B, 1]


_HEAD_REGISTRY = {
    "linear": LinearRankingHead,
    "mlp": MLPRankingHead,
}


def build_ranking_head(head_type: str = "mlp", **kwargs) -> nn.Module:
    """Factory for ranking head variants."""
    cls = _HEAD_REGISTRY.get(head_type)
    if cls is None:
        raise ValueError(f"Unknown head_type '{head_type}'. Choose from {list(_HEAD_REGISTRY)}")
    return cls(**kwargs)


@MODELS.register_module()
class SiameseOrdinalCLIP(nn.Module):
    """Siamese ranking model wrapping a frozen OrdinalCLIP / Baseline backbone.

    forward():
        Accepts (images_a, images_b), returns ranking_logits + per-image logits.
    forward_single():
        Accepts images, returns classification logits (for val/test eval).
    """

    def __init__(
        self,
        backbone: nn.Module,
        ranking_head_cfg: dict,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.ranking_head = build_ranking_head(**ranking_head_cfg)

        self.embed_dims = backbone.embed_dims
        self.num_ranks = backbone.num_ranks

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            logger.info("Backbone frozen for Siamese training.")

        trainable = sum(p.numel() for p in self.ranking_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"SiameseOrdinalCLIP: ranking_head {trainable:,} params, total {total:,} params")

    def forward(
        self,
        images_a: torch.Tensor,
        images_b: torch.Tensor,
    ):
        """Pairwise forward.

        Returns:
            ranking_logits: [B, 1]
            logits_a:       [B, num_ranks]
            logits_b:       [B, num_ranks]
        """
        logits_a, feat_a, _ = self.backbone(images_a)  # [B, num_ranks], [B, D], _
        logits_b, feat_b, _ = self.backbone(images_b)  # [B, num_ranks], [B, D], _

        feat_diff = feat_a - feat_b  # [B, D]
        ranking_logits = self.ranking_head(feat_diff)   # [B, 1]

        return ranking_logits, logits_a, logits_b

    def forward_single(self, images: torch.Tensor):
        """Single-image forward for val / test classification."""
        return self.backbone(images)

    def train(self, mode: bool = True):
        """Override to keep frozen backbone in eval mode."""
        super().train(mode)
        # If backbone is frozen, always keep it in eval (BatchNorm stats frozen)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

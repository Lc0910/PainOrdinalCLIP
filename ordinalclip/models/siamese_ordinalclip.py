"""Siamese OrdinalCLIP — aligned to Fabio (2025) dissertation §5.3.

Architecture (Stage 2, backbone frozen):
  image → OrdinalCLIP backbone → f [B, D]
                                    ↓
                             SharedMLP (2-layer)
                             D → hidden → out  (e [B, out])
                            /                    \\
              RegressionHead               ConcatRankingHead
              out → 1 → sigmoid            [ei ∥ ej] (2*out) → 1
              ŷ = (K-1) · σ(z) ∈ [0,K-1]  s_ij (signed ranking score)

Training objective (SiameseRunner):
  L = L_mse + λ · L_hinge
  L_mse   = MSE(ŷ_a, y_a) + MSE(ŷ_b, y_b)
  L_hinge = mean max(0,  |y_a - y_b| · margin_scale  −  η · s_ab)
  η = sign(y_a − y_b)  (+1 if a ranks higher than b)

Key design decisions vs prior implementation:
  - SharedMLP: both heads share the same compact embedding; gradients
    from both MSE and hinge update the same MLP (≈ Fabio §5.1.1).
  - Concatenation input [ei ∥ ej]: ranking head sees more information
    than with the difference (fi − fj). (Fabio §5.3.1, §5.1.1).
  - Regression + hinge instead of classification + BCE: continuous
    regression with adaptive margin is more natural for ordinal data.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ordinalclip.utils import get_logger

from .builder import MODELS

logger = get_logger(__name__)


# ================================================================
#  Shared MLP
# ================================================================

class SharedMLP(nn.Module):
    """2-layer shared projection MLP.

    Projects backbone features into a compact embedding that is
    passed to both the regression head and the ranking head.

    Input:  f  [B, embed_dims]
    Output: e  [B, out_dims]
    """

    def __init__(
        self,
        embed_dims: int,
        hidden_dims: int = 256,
        out_dims: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(embed_dims, hidden_dims),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden_dims, out_dims),
            nn.ReLU(inplace=True),
        ]
        self.mlp = nn.Sequential(*layers)
        self.out_dims = out_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # [B, out_dims]


# ================================================================
#  Regression Head
# ================================================================

class RegressionHead(nn.Module):
    """Single linear layer with sigmoid rescaling.

    Output: ŷ = (num_ranks - 1) · σ(z)  ∈ (0, num_ranks - 1)

    For K=5 pain classes (0..4): ŷ ∈ (0, 4), round(ŷ) gives the class.
    For K=2 binary (0..1):       ŷ ∈ (0, 1), ≈ probability of class 1.
    """

    def __init__(self, in_dims: int, num_ranks: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dims, 1)
        self.num_ranks = num_ranks

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        z = self.linear(e)  # [B, 1]
        score = (self.num_ranks - 1) * torch.sigmoid(z).squeeze(1)  # [B]
        return score


# ================================================================
#  Concat Ranking Head
# ================================================================

class ConcatRankingHead(nn.Module):
    """Pairwise ranking head operating on concatenated embeddings.

    Receives [ei ∥ ej] ∈ R^{2·in_dims} and outputs a signed score s_ij.
    Positive s_ij indicates image i ranks higher than j.

    Linear variant (paper default): 2*in_dims → 1.
    MLP variant: 2*in_dims → hidden → hidden//2 → 1.
    """

    def __init__(
        self,
        in_dims: int,
        head_type: str = "linear",
        hidden_dims: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        concat_dims = 2 * in_dims
        if head_type == "linear":
            self.net = nn.Linear(concat_dims, 1)
        elif head_type == "mlp":
            layers: list[nn.Module] = [
                nn.Linear(concat_dims, hidden_dims),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers += [
                nn.Linear(hidden_dims, hidden_dims // 2),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dims // 2, 1))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(
                f"Unknown head_type '{head_type}'. Choose from ['linear', 'mlp']"
            )
        self.head_type = head_type

    def forward(self, concat_feat: torch.Tensor) -> torch.Tensor:
        return self.net(concat_feat)  # [B, 1]


# ================================================================
#  Siamese OrdinalCLIP (main model)
# ================================================================

@MODELS.register_module()
class SiameseOrdinalCLIP(nn.Module):
    """Siamese ranking model wrapping a frozen OrdinalCLIP backbone.

    Implements the Fabio (2025) architecture:
      - SharedMLP: compact shared representation
      - RegressionHead: continuous ordinal score ŷ ∈ [0, K-1]
      - ConcatRankingHead: pairwise score from concatenated embeddings

    forward(images_a, images_b):
        Training — returns (ranking_logit, reg_score_a, reg_score_b)

    forward_single(images):
        Val/Test — returns (logits, image_features, reg_score)
        where `logits` is OrdinalCLIP's softmax classification (for ablation),
        `reg_score` is the regression head output (primary metric).
    """

    def __init__(
        self,
        backbone: nn.Module,
        shared_mlp_cfg: dict,
        ranking_head_cfg: dict,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dims = backbone.embed_dims
        self.num_ranks = backbone.num_ranks

        # Shared MLP
        self.shared_mlp = SharedMLP(embed_dims=self.embed_dims, **shared_mlp_cfg)
        mlp_out = self.shared_mlp.out_dims

        # Regression head
        self.regression_head = RegressionHead(in_dims=mlp_out, num_ranks=self.num_ranks)

        # Concat Ranking head
        self.ranking_head = ConcatRankingHead(in_dims=mlp_out, **ranking_head_cfg)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            logger.info("Backbone frozen for Siamese training.")

        n_mlp = sum(p.numel() for p in self.shared_mlp.parameters())
        n_reg = sum(p.numel() for p in self.regression_head.parameters())
        n_rank = sum(p.numel() for p in self.ranking_head.parameters())
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"SiameseOrdinalCLIP: shared_mlp={n_mlp:,}, "
            f"regression_head={n_reg:,}, ranking_head={n_rank:,}, "
            f"total={n_total:,} params"
        )

    def forward(
        self,
        images_a: torch.Tensor,
        images_b: torch.Tensor,
    ):
        """Pairwise forward for training.

        Args:
            images_a: [B, 3, H, W]
            images_b: [B, 3, H, W]

        Returns:
            ranking_logit: [B, 1]  signed score (positive → a > b)
            reg_score_a:   [B]     continuous ordinal score ∈ (0, K-1)
            reg_score_b:   [B]
        """
        _, feat_a, _ = self.backbone(images_a)  # [B, D]
        _, feat_b, _ = self.backbone(images_b)  # [B, D]

        e_a = self.shared_mlp(feat_a)  # [B, out]
        e_b = self.shared_mlp(feat_b)  # [B, out]

        reg_score_a = self.regression_head(e_a)  # [B]
        reg_score_b = self.regression_head(e_b)  # [B]

        concat_feat = torch.cat([e_a, e_b], dim=-1)  # [B, 2*out]
        ranking_logit = self.ranking_head(concat_feat)  # [B, 1]

        return ranking_logit, reg_score_a, reg_score_b

    def forward_single(self, images: torch.Tensor):
        """Single-image forward for val/test.

        Returns:
            logits:         [B, K]  OrdinalCLIP classification logits (for ablation)
            image_features: [B, D]  raw CLIP features (for anchor computation)
            reg_score:      [B]     regression head output — primary metric
        """
        logits, feat, _ = self.backbone(images)  # [B, K], [B, D], _
        if feat is not None:
            e = self.shared_mlp(feat)    # [B, out]
            reg_score = self.regression_head(e)  # [B]
        else:
            reg_score = None
        return logits, feat, reg_score

    def train(self, mode: bool = True):
        """Keep frozen backbone in eval mode to preserve BatchNorm stats."""
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

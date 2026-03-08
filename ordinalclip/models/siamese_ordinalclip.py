"""Siamese OrdinalCLIP — aligned to Fabio (2025) dissertation §5.3.

Architecture (Stage 2, backbone frozen):
  image → OrdinalCLIP backbone → f [B, D]
                                    ↓
                          (optional) concat AU features
                            [f ∥ au] [B, D + au_dim]
                                    ↓
                             SharedMLP (2-layer)
                             D(+au) → hidden → out  (e [B, out])
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
  - AU fusion: optional early concatenation of AU features before SharedMLP.
    Controlled by au_cfg.enabled; when disabled, zero overhead.
"""
from __future__ import annotations

from typing import Optional, Tuple

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
        au_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dims = backbone.embed_dims
        self.num_ranks = backbone.num_ranks

        # AU fusion config
        self.au_cfg = au_cfg or {"enabled": False}
        self._au_enabled = self.au_cfg.get("enabled", False)
        au_dim = self.au_cfg.get("au_dim", 0) if self._au_enabled else 0

        # Fail-fast: enabled + au_dim<=0 is an invalid configuration.
        # SharedMLP would be built with embed_dims+0 but _fuse_au would still
        # be called, producing a shape mismatch.
        if self._au_enabled and au_dim <= 0:
            raise ValueError(
                f"au_cfg.enabled=True but au_dim={au_dim} (must be > 0).  "
                f"Set au_dim to the number of AU columns (17 for all, 8 for pain)."
            )

        # AU preprocessing layers (LayerNorm + Dropout to align scale with CLIP features)
        # Always initialize to None for safe hasattr checks
        self.au_norm: Optional[nn.LayerNorm] = None
        self.au_dropout: Optional[nn.Dropout] = None
        if au_dim > 0:
            self.au_norm = nn.LayerNorm(au_dim)
            self.au_dropout = nn.Dropout(self.au_cfg.get("au_dropout", 0.1))
            logger.info(f"AU fusion enabled: au_dim={au_dim}, dropout={self.au_cfg.get('au_dropout', 0.1)}")

        # Shared MLP — input dim = CLIP embed_dims + au_dim
        mlp_in_dim = self.embed_dims + au_dim
        self.shared_mlp = SharedMLP(embed_dims=mlp_in_dim, **shared_mlp_cfg)
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
        n_au = sum(p.numel() for p in self.au_norm.parameters()) if au_dim > 0 else 0
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"SiameseOrdinalCLIP: shared_mlp={n_mlp:,} (in={mlp_in_dim}), "
            f"regression_head={n_reg:,}, ranking_head={n_rank:,}, "
            f"au_layers={n_au:,}, total={n_total:,} params"
        )

    def _fuse_au(self, clip_feat: torch.Tensor, au_feat: torch.Tensor) -> torch.Tensor:
        """Fuse CLIP features with AU features via early concatenation.

        Args:
            clip_feat: [B, D] CLIP image features
            au_feat:   [B, au_dim] AU intensity features

        Returns:
            fused: [B, D + au_dim]
        """
        if self.au_norm is None or self.au_dropout is None:
            raise RuntimeError(
                "AU layers not initialized. Ensure au_cfg.au_dim > 0 "
                "when au_cfg.enabled=True."
            )
        au_feat = self.au_norm(au_feat.float())    # [B, au_dim]  normalize scale
        au_feat = self.au_dropout(au_feat)         # [B, au_dim]  regularization
        return torch.cat([clip_feat, au_feat], dim=-1)  # [B, D + au_dim]

    def forward(
        self,
        images_a: torch.Tensor,
        images_b: torch.Tensor,
        au_a: Optional[torch.Tensor] = None,
        au_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pairwise forward for training.

        Args:
            images_a: [B, 3, H, W]
            images_b: [B, 3, H, W]
            au_a:     [B, au_dim] or None  — AU features for image A
            au_b:     [B, au_dim] or None  — AU features for image B

        Returns:
            ranking_logit: [B, 1]  signed score (positive → a > b)
            reg_score_a:   [B]     continuous ordinal score ∈ (0, K-1)
            reg_score_b:   [B]
        """
        _, feat_a, _ = self.backbone(images_a)  # [B, D]
        _, feat_b, _ = self.backbone(images_b)  # [B, D]

        # AU fusion: early concatenation before SharedMLP
        if self._au_enabled and au_a is not None and au_b is not None:
            feat_a = self._fuse_au(feat_a, au_a)  # [B, D + au_dim]
            feat_b = self._fuse_au(feat_b, au_b)  # [B, D + au_dim]

        e_a = self.shared_mlp(feat_a)  # [B, out]
        e_b = self.shared_mlp(feat_b)  # [B, out]

        reg_score_a = self.regression_head(e_a)  # [B]
        reg_score_b = self.regression_head(e_b)  # [B]

        concat_feat = torch.cat([e_a, e_b], dim=-1)  # [B, 2*out]
        ranking_logit = self.ranking_head(concat_feat)  # [B, 1]

        return ranking_logit, reg_score_a, reg_score_b

    def forward_single(
        self,
        images: torch.Tensor,
        au_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Single-image forward for val/test.

        Args:
            images:  [B, 3, H, W]
            au_feat: [B, au_dim] or None — AU features

        Returns:
            logits:         [B, K]  OrdinalCLIP classification logits (for ablation)
            image_features: [B, D]  raw CLIP features (for anchor computation)
            reg_score:      [B]     regression head output — primary metric
        """
        logits, feat, _ = self.backbone(images)  # [B, K], [B, D], _
        if feat is not None:
            # AU fusion before SharedMLP
            fused = feat
            if self._au_enabled and au_feat is not None:
                fused = self._fuse_au(feat, au_feat)  # [B, D + au_dim]
            e = self.shared_mlp(fused)    # [B, out]
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

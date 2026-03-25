"""AU-Guided Attention Fusion (AGAF) module.

Novel multi-modal fusion mechanism that uses per-AU tokenization and
cross-attention to selectively incorporate Action Unit information into
visual features for pain intensity estimation.

Key innovations:
1. Per-AU Tokenization: Each AU intensity becomes a separate attention token,
   allowing the model to learn which AUs are informative per sample.
2. Gated Residual: A learned sigmoid gate controls how much AU information
   flows into the final representation, gracefully degrading to visual-only
   when AU features are noisy.
3. Dimension-preserving: Output has the same dimension as visual input,
   making it a drop-in replacement for concatenation-based fusion.

Architecture:
    visual_feat [B, D]          au_feat [B, A]
        |                           |
        |                    Per-AU Embedding: each AU_i -> [B, D_au]
        |                    -> AU tokens [B, N_au, D_au]
        |                           |
        +------ Cross-Attention ----+
        |   Q = Linear(visual_feat) [B, 1, D_au]
        |   K = AU tokens           [B, N_au, D_au]
        |   V = AU tokens           [B, N_au, D_au]
        |            |
        |   attn_out [B, D]
        |            |
        +--- Gate = sigma(W * [visual || attn_out]) ---+
        |                                               |
        fused = gate * attn_out + (1 - gate) * visual_feat
        |
        [B, D]
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AUGuidedAttentionFusion(nn.Module):
    """AU-Guided Attention Fusion (AGAF).

    Fuses visual features with AU (Action Unit) features through cross-attention
    and gated residual connection.

    Args:
        visual_dim: Dimension of visual feature vector (D).
        au_dim: Number of AU features (A, e.g. 8 or 17).
        hidden_dim: Hidden dimension for AU token embeddings (D_au).
            Defaults to min(visual_dim, 128).
        num_heads: Number of attention heads for cross-attention.
        dropout: Dropout rate for attention and gate.
    """

    def __init__(
        self,
        visual_dim: int,
        au_dim: int,
        hidden_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.au_dim = au_dim
        self.hidden_dim = hidden_dim or min(visual_dim, 128)
        self.num_heads = num_heads

        assert self.hidden_dim % num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        # Per-AU tokenization: each AU scalar -> D_au vector via shared projection
        # + learnable positional embedding per AU identity
        self.au_token_proj = nn.Linear(1, self.hidden_dim)  # shared projection per AU scalar
        self.au_pos_embed = nn.Parameter(
            torch.randn(1, au_dim, self.hidden_dim) * 0.02
        )  # [1, N_au, D_au] positional embedding per AU identity

        # Layer norm for AU tokens
        self.au_ln = nn.LayerNorm(self.hidden_dim)

        # Query projection: visual_feat [B, D] -> [B, 1, D_au]
        self.query_proj = nn.Linear(visual_dim, self.hidden_dim)

        # Cross-attention: Q from visual, K/V from AU tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_ln = nn.LayerNorm(self.hidden_dim)

        # Project attention output back to visual dimension
        self.out_proj = nn.Linear(self.hidden_dim, visual_dim)

        # Gating mechanism: sigmoid gate over [visual || attn_out]
        self.gate = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual_feat: torch.Tensor,
        au_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse visual and AU features via cross-attention gating.

        Args:
            visual_feat: [B, D] visual features from FER encoder.
            au_feat: [B, A] AU intensity values.

        Returns:
            fused: [B, D] fused features (same dimension as visual input).
        """
        B = visual_feat.size(0)

        # --- Per-AU Tokenization ---
        # Each AU scalar -> a D_au dimensional token
        # au_feat: [B, A] -> expand each AU to [B, A, 1] -> project to [B, A, D_au]
        au_expanded = au_feat.unsqueeze(-1)  # [B, A, 1]
        au_tokens = self.au_token_proj(au_expanded)  # [B, A, D_au]
        au_tokens = au_tokens + self.au_pos_embed  # [B, A, D_au] + positional identity
        au_tokens = self.au_ln(au_tokens)  # [B, A, D_au]

        # --- Cross-Attention ---
        # Query from visual features: [B, D] -> [B, 1, D_au]
        query = self.query_proj(visual_feat).unsqueeze(1)  # [B, 1, D_au]

        # Cross-attention: Q=[B,1,D_au], K=V=[B,A,D_au] -> [B,1,D_au]
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=au_tokens,
            value=au_tokens,
        )  # attn_out: [B, 1, D_au], attn_weights: [B, 1, A]

        attn_out = self.attn_ln(attn_out)  # [B, 1, D_au]
        attn_out = attn_out.squeeze(1)  # [B, D_au]

        # Project back to visual dimension
        attn_feat = self.out_proj(attn_out)  # [B, D]
        attn_feat = self.dropout(attn_feat)  # [B, D]

        # --- Gated Residual ---
        gate_input = torch.cat([visual_feat, attn_feat], dim=1)  # [B, 2D]
        gate_value = self.gate(gate_input)  # [B, D] in (0, 1)

        fused = gate_value * attn_feat + (1.0 - gate_value) * visual_feat  # [B, D]

        return fused

    def get_attention_weights(
        self,
        visual_feat: torch.Tensor,
        au_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Get per-AU attention weights for interpretability.

        Returns:
            attn_weights: [B, 1, A] attention weights over AUs (averaged over heads).
        """
        B = visual_feat.size(0)
        au_expanded = au_feat.unsqueeze(-1)  # [B, A, 1]
        au_tokens = self.au_token_proj(au_expanded)  # [B, A, D_au]
        au_tokens = au_tokens + self.au_pos_embed
        au_tokens = self.au_ln(au_tokens)

        query = self.query_proj(visual_feat).unsqueeze(1)  # [B, 1, D_au]

        _, attn_weights = self.cross_attn(
            query=query, key=au_tokens, value=au_tokens,
        )  # [B, 1, A]

        return attn_weights


class ConcatFusion(nn.Module):
    """Simple concatenation fusion (existing baseline).

    Concatenates visual and AU features then projects to visual_dim.
    Used for ablation comparison.
    """

    def __init__(
        self,
        visual_dim: int,
        au_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.au_norm = nn.LayerNorm(au_dim)
        self.au_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(visual_dim + au_dim, visual_dim)

    def forward(
        self,
        visual_feat: torch.Tensor,
        au_feat: torch.Tensor,
    ) -> torch.Tensor:
        """visual_feat: [B, D], au_feat: [B, A] -> fused: [B, D]"""
        au = self.au_dropout(self.au_norm(au_feat))  # [B, A]
        concat = torch.cat([visual_feat, au], dim=1)  # [B, D+A]
        return self.proj(concat)  # [B, D]


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation (FiLM) fusion.

    AU features condition on visual features via learned scale and shift:
        fused = gamma(au) * visual + beta(au)

    Used for ablation comparison.
    """

    def __init__(
        self,
        visual_dim: int,
        au_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.au_norm = nn.LayerNorm(au_dim)
        self.gamma_net = nn.Sequential(
            nn.Linear(au_dim, visual_dim),
            nn.Dropout(dropout),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(au_dim, visual_dim),
            nn.Dropout(dropout),
        )
        # Init near-zero so initial behavior is close to identity (gamma~=1, beta~=0)
        nn.init.zeros_(self.gamma_net[0].weight)
        nn.init.zeros_(self.gamma_net[0].bias)
        nn.init.zeros_(self.beta_net[0].weight)
        nn.init.zeros_(self.beta_net[0].bias)

    def forward(
        self,
        visual_feat: torch.Tensor,
        au_feat: torch.Tensor,
    ) -> torch.Tensor:
        """visual_feat: [B, D], au_feat: [B, A] -> fused: [B, D]"""
        au = self.au_norm(au_feat)  # [B, A]
        gamma = 1.0 + self.gamma_net(au)  # [B, D], centered at 1
        beta = self.beta_net(au)  # [B, D]
        return gamma * visual_feat + beta  # [B, D]


def build_au_fusion(
    fusion_type: str,
    visual_dim: int,
    au_dim: int,
    **kwargs,
) -> nn.Module:
    """Factory function for AU fusion modules.

    Args:
        fusion_type: One of 'agaf', 'concat', 'film'.
        visual_dim: Visual feature dimension.
        au_dim: Number of AU features.

    Returns:
        Fusion module with interface: forward(visual_feat, au_feat) -> fused.
    """
    fusion_map = {
        "agaf": AUGuidedAttentionFusion,
        "concat": ConcatFusion,
        "film": FiLMFusion,
    }

    if fusion_type not in fusion_map:
        raise ValueError(
            f"Unknown fusion_type: {fusion_type}. Available: {list(fusion_map.keys())}"
        )

    return fusion_map[fusion_type](visual_dim=visual_dim, au_dim=au_dim, **kwargs)

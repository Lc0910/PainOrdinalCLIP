"""Ordinal regression loss functions.

CORAL (Consistent Rank Logits):
    Cao, W., Mirjalili, V., Raschka, S. (2020). "Rank Consistent Ordinal
    Regression for Neural Networks with Application to Age Estimation."
    Pattern Recognition Letters.

    Instead of K-class softmax, CORAL uses K-1 binary classifiers:
        P(Y > k | x) for k = 0, 1, ..., K-2

    This naturally encodes the ordinal structure: if P(Y > 3) is high,
    P(Y > 2) should also be high.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CORALLoss(nn.Module):
    """CORAL (Consistent Rank Logits) ordinal regression loss.

    Treats K-class ordinal regression as K-1 binary classification problems:
    for each threshold k, predict P(Y > k | x).

    The model should output K-1 logits (one per threshold), and the loss
    is the sum of binary cross-entropy losses across all thresholds.

    Args:
        num_ranks: Number of ordinal ranks (K). Loss uses K-1 thresholds.
        reduction: 'mean' or 'sum' or 'none'.
    """

    def __init__(self, num_ranks: int, reduction: str = "mean") -> None:
        super().__init__()
        self.num_ranks = num_ranks
        self.num_thresholds = num_ranks - 1
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CORAL loss.

        Args:
            logits: [B, K-1] raw logits for each threshold.
                If [B, K] is passed (from standard softmax head), the last
                column is dropped and cumulative logits are computed.
            targets: [B] integer labels in {0, 1, ..., K-1}.

        Returns:
            Scalar loss (reduced) or [B] per-sample loss.
        """
        B = logits.size(0)

        # Handle K-dim logits from softmax-style models
        if logits.size(1) == self.num_ranks:
            # Convert K-class logits to K-1 cumulative logits
            # Use cumulative sum approach: logit_k = sum(logits[:k+1])
            logits = self._softmax_to_cumulative(logits)  # [B, K-1]

        assert logits.size(1) == self.num_thresholds, (
            f"Expected {self.num_thresholds} threshold logits, got {logits.size(1)}"
        )

        # Binary targets: target_k = 1 if Y > k, else 0
        # For target=3, K=5: binary_targets = [1, 1, 1, 0]
        levels = torch.arange(self.num_thresholds, device=targets.device)  # [K-1]
        binary_targets = (targets.unsqueeze(1) > levels.unsqueeze(0)).float()  # [B, K-1]

        # Binary cross-entropy for each threshold
        loss_per_threshold = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction="none"
        )  # [B, K-1]

        # Sum across thresholds per sample
        loss_per_sample = loss_per_threshold.sum(dim=1)  # [B]

        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample

    def _softmax_to_cumulative(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert K-class softmax logits to K-1 cumulative threshold logits.

        Uses the relationship: P(Y > k) = sum_{j=k+1}^{K-1} P(Y = j)
        In logit space, we compute: cumlogit_k = log(P(Y>k) / P(Y<=k))

        Args:
            logits: [B, K] standard class logits.

        Returns:
            cumulative_logits: [B, K-1] threshold logits.
        """
        probs = F.softmax(logits, dim=1)  # [B, K]
        # Cumulative probability: P(Y > k) = 1 - CDF(k)
        cdf = probs.cumsum(dim=1)  # [B, K]
        # P(Y > k) for k = 0..K-2
        prob_greater = 1.0 - cdf[:, :-1]  # [B, K-1]
        # Clamp for numerical stability
        prob_greater = prob_greater.clamp(min=1e-7, max=1.0 - 1e-7)
        # Convert to logits
        cumulative_logits = torch.log(prob_greater / (1.0 - prob_greater))  # [B, K-1]
        return cumulative_logits

    @staticmethod
    def predict(logits: torch.Tensor, num_ranks: int) -> torch.Tensor:
        """Predict ordinal rank from CORAL threshold logits.

        Count the number of thresholds where P(Y > k) > 0.5 (i.e. logit > 0).

        Args:
            logits: [B, K-1] threshold logits, or [B, K] softmax logits.
            num_ranks: K.

        Returns:
            predictions: [B] predicted ranks in {0, ..., K-1}.
        """
        if logits.size(1) == num_ranks:
            # Softmax logits: convert to cumulative first
            probs = F.softmax(logits, dim=1)
            cdf = probs.cumsum(dim=1)[:, :-1]  # [B, K-1]
            predictions = (cdf < 0.5).sum(dim=1)  # [B]
        else:
            # Threshold logits: count exceedances
            predictions = (logits > 0).sum(dim=1)  # [B]
        return predictions

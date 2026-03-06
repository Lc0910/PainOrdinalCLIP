"""Siamese Runner for Stage 2 pairwise ranking training.

Aligned to Fabio (2025) dissertation §5.3 — Siamese OrdinalCLIP:

Architecture:
  OrdinalCLIP backbone (frozen) → SharedMLP → RegressionHead + ConcatRankingHead

Training loss (§5.1.1.2):
  L = L_mse + λ · L_hinge
  L_mse   = MSE(ŷ_a, y_a) + MSE(ŷ_b, y_b)   (regression, both images)
  L_hinge = mean max(0, |y_a-y_b|·margin_scale − η·s_ab)  (adaptive margin)
  η = sign(y_a − y_b)  (+1 if a ranks higher)

Val/Test:
  - Regression-based prediction: ŷ = RegressionHead(SharedMLP(f))
  - OrdinalCLIP classification logits also logged for ablation
  - Video-level aggregation same as Stage 1 Runner
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ordinalclip.models import MODELS
from ordinalclip.models.siamese_ordinalclip import SiameseOrdinalCLIP
from ordinalclip.utils.logging import get_logger

from .optim import build_lr_scheduler, build_optimizer

logger = get_logger(__name__)


class SiameseRunner(pl.LightningModule):
    """PyTorch Lightning module for Siamese OrdinalCLIP Stage 2 training.

    Implements Fabio (2025) §5.3: frozen OrdinalCLIP backbone + SharedMLP
    + RegressionHead + ConcatRankingHead, trained with MSE + adaptive hinge.
    """

    def __init__(
        self,
        backbone_cfg: dict,
        shared_mlp_cfg: dict,
        ranking_head_cfg: dict,
        output_dir: str,
        optimizer_and_scheduler_cfg: dict,
        load_weights_cfg: dict,
        seed: int,
        loss_weights: Optional[Dict] = None,
        freeze_backbone: bool = True,
        ckpt_path: str = "",
        anchor_inference_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        if loss_weights is None:
            loss_weights = {
                "ranking_loss": 0.5,    # λ (multiplier on hinge loss)
                "mse_loss": 1.0,        # weight on MSE regression loss
                "margin_scale": 1.0,    # adaptive margin = |yi-yj| * margin_scale
            }

        # --- 1. Build backbone from config (OrdinalCLIP / Baseline) ---
        backbone = MODELS.build(backbone_cfg)

        # --- 2. Build Siamese model ---
        self.module = SiameseOrdinalCLIP(
            backbone=backbone,
            shared_mlp_cfg=shared_mlp_cfg,
            ranking_head_cfg=ranking_head_cfg,
            freeze_backbone=freeze_backbone,
        )

        # --- Output & logger (must be initialized before _load_backbone_weights) ---
        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path

        # --- 3. Load backbone weights from Stage 1 checkpoint ---
        self._load_backbone_weights(**load_weights_cfg)

        # --- Loss functions ---
        self.mse_loss_func = nn.MSELoss()
        self.ce_loss_func = nn.CrossEntropyLoss()   # kept for ablation logging
        self.loss_weights = loss_weights
        self._freeze_backbone = freeze_backbone

        # --- Metrics helpers ---
        self.num_ranks = self.module.num_ranks
        self.register_buffer(
            "rank_output_value_array",
            torch.arange(0, self.num_ranks).float(),
            persistent=False,
        )

        # --- Epoch-level ranking metric accumulators ---
        self._train_ranking_scores: List[torch.Tensor] = []
        self._train_ranking_labels: List[torch.Tensor] = []

        # --- Anchor-based ranking inference (disabled by default) ---
        self._anchor_cfg: Dict = anchor_inference_cfg or {"enabled": False}
        self._anchors: Optional[Dict[int, torch.Tensor]] = None

    # ================================================================
    #  Forward
    # ================================================================

    def forward(self, images_a: torch.Tensor, images_b: torch.Tensor):
        """Pairwise forward for training."""
        return self.module(images_a, images_b)

    def forward_single(self, images: torch.Tensor):
        """Single-image forward for val/test."""
        return self.module.forward_single(images)

    # ================================================================
    #  Training
    # ================================================================

    def training_step(self, batch, batch_idx):
        img_a, img_b, pair_label, rank_a, rank_b = batch
        # img_a/img_b: [B, 3, H, W]
        # pair_label: [B]  (1 if rank_a > rank_b, 0 otherwise)
        # rank_a/rank_b: [B]  (integer class labels)

        ranking_logit, reg_score_a, reg_score_b = self.module(img_a, img_b)
        # ranking_logit: [B, 1]  signed score (positive → a > b)
        # reg_score_a/b: [B]     continuous ŷ ∈ (0, K-1)

        # --- MSE regression loss (both images) ---
        y_a = rank_a.float()  # [B]
        y_b = rank_b.float()  # [B]
        mse_loss = self.mse_loss_func(reg_score_a, y_a) + \
                   self.mse_loss_func(reg_score_b, y_b)

        # --- Adaptive margin hinge ranking loss ---
        hinge_loss = self._adaptive_hinge_loss(ranking_logit, rank_a, rank_b)

        loss = (
            self.loss_weights.get("mse_loss", 1.0) * mse_loss
            + self.loss_weights.get("ranking_loss", 0.5) * hinge_loss
        )

        # --- Per-batch ranking accuracy ---
        with torch.no_grad():
            s = ranking_logit.squeeze(1)  # [B]
            eta = (rank_a.float() - rank_b.float()).sign()  # [B]
            pairwise_correct = ((s * eta) > 0).float()
            pairwise_acc = pairwise_correct.mean()

            # Accumulate signed scores + pair labels for epoch-level AUC
            # (convert to prob-like: positive s → higher P(a>b))
            self._train_ranking_scores.append(s.detach().cpu())
            self._train_ranking_labels.append(pair_label.cpu())

            # Regression accuracy (round ŷ to nearest class)
            reg_acc_a = (torch.round(reg_score_a) == y_a).float().mean()
            reg_acc_b = (torch.round(reg_score_b) == y_b).float().mean()

        self.log("train_loss",        loss,         on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mse_loss",    mse_loss,     on_step=True, on_epoch=True)
        self.log("train_hinge_loss",  hinge_loss,   on_step=True, on_epoch=True)
        self.log("train_pair_acc",    pairwise_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_reg_acc_a",   reg_acc_a,    on_step=False, on_epoch=True)
        self.log("train_reg_acc_b",   reg_acc_b,    on_step=False, on_epoch=True)

        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        """Epoch-level pairwise AUC."""
        if self._train_ranking_scores:
            all_scores = torch.cat(self._train_ranking_scores)   # [N]
            all_labels = torch.cat(self._train_ranking_labels)   # [N]
            # Convert signed scores to probabilities for AUC
            probs = torch.sigmoid(all_scores)
            auc = self._compute_binary_auc(probs, all_labels)
            self.log("train_pair_auc", auc, on_epoch=True, prog_bar=True)

        self._train_ranking_scores.clear()
        self._train_ranking_labels.clear()

    # ================================================================
    #  Adaptive Hinge Loss
    # ================================================================

    def _adaptive_hinge_loss(
        self,
        ranking_logit: torch.Tensor,
        rank_a: torch.Tensor,
        rank_b: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptive margin hinge ranking loss (Fabio §5.1.1.2).

        L_hinge = mean max(0, |y_a - y_b| * margin_scale − η * s_ab)
        η = sign(y_a − y_b):  +1 if a > b,  −1 if a < b

        The margin scales linearly with the label distance:
        adjacent classes (diff=1) get a small margin, distant classes
        (diff=4 for 5-class) get a large margin, encouraging stronger
        separation when the ground-truth difference is pronounced.

        Args:
            ranking_logit: [B, 1]
            rank_a:        [B]  integer labels
            rank_b:        [B]  integer labels

        Returns:
            scalar hinge loss
        """
        s = ranking_logit.squeeze(1)  # [B]
        rank_diff = (rank_a.float() - rank_b.float())  # [B]
        assert (rank_diff != 0).all(), (
            "Hinge loss undefined for equal-label pairs. "
            "PairwiseDataset should guarantee cross-class sampling."
        )
        eta = torch.sign(rank_diff)                    # [B]: +1 or -1
        margin = torch.abs(rank_diff) * self.loss_weights.get("margin_scale", 1.0)
        return torch.relu(margin - eta * s).mean()

    # ================================================================
    #  Validation / Test  (single-image classification + regression)
    # ================================================================

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch)

    def _eval_step(self, batch):
        """Regression + optional anchor-based ranking evaluation."""
        x, y = batch[0], batch[1]  # [B, 3, H, W], [B]

        logits, image_features, reg_score = self.module.forward_single(x)
        # logits:         [B, K]  — OrdinalCLIP classification (ablation)
        # image_features: [B, D]  — raw CLIP features (anchor inference)
        # reg_score:      [B]     — regression head output (primary metric)

        # CE loss on OrdinalCLIP logits (for ablation comparison)
        ce_loss = self.ce_loss_func(logits, y)

        # Primary: regression-based metrics
        metrics_reg = self.compute_regression_metrics(reg_score, y)

        # Ablation: classification-based metrics (exp / max)
        metrics_exp = self.compute_per_example_metrics(logits, y, "exp")
        metrics_max = self.compute_per_example_metrics(logits, y, "max")

        outputs = {
            "loss": metrics_reg["mae_reg_metric"].mean(),
            "ce_loss": ce_loss,
            **metrics_reg,
            **metrics_exp,
            **metrics_max,
        }

        # Anchor-based ranking inference
        if self._anchors is not None and image_features is not None:
            rank_metrics = self._compute_rank_predictions(image_features, y)
            p_rank = rank_metrics.pop("_p_rank")
            outputs.update(rank_metrics)

            p_cls = F.softmax(logits, dim=-1)
            ens_metrics = self._compute_ensemble_predictions(p_cls, p_rank, y)
            outputs.update(ens_metrics)

        if len(batch) > 2:
            outputs["_paths"] = batch[2]
            outputs["_targets"] = batch[1]
        return outputs

    def validation_epoch_end(self, outputs) -> None:
        self._eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self._eval_epoch_end(outputs, "test")

    # ================================================================
    #  Epoch-end aggregation
    # ================================================================

    loggings_suffix = {"metric", "loss"}

    def _valid_key(self, key: str) -> bool:
        return any(key.endswith(suffix) for suffix in self.loggings_suffix)

    def _eval_epoch_end(self, outputs, run_type: str) -> None:
        """Frame-level stats + video-level aggregation."""
        stats: Dict[str, list] = defaultdict(list)
        for _outputs in outputs:
            for k, v in _outputs.items():
                if self._valid_key(k):
                    stats[k].append(v)

        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            is_primary = (k == "mae_reg_metric")
            self.log(
                f"{run_type}_{k}", stats[k],
                on_step=False, on_epoch=True,
                prog_bar=is_primary, logger=True,
            )

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

        self._video_level_aggregation(outputs, run_type)

    def _video_level_aggregation(self, outputs, run_type: str) -> None:
        """Aggregate frame predictions to video-level results."""
        has_rank = any("predict_y_rank" in o for o in outputs)
        has_ens = any("predict_y_ens" in o for o in outputs)

        all_paths: List[str] = []
        all_targets: List[torch.Tensor] = []
        all_pred_reg: List[torch.Tensor] = []
        all_pred_exp: List[torch.Tensor] = []
        all_pred_max: List[torch.Tensor] = []
        all_pred_rank: List[torch.Tensor] = []
        all_pred_ens: List[torch.Tensor] = []

        for _outputs in outputs:
            if "_paths" not in _outputs:
                continue
            all_paths.extend(_outputs["_paths"])
            all_targets.append(_outputs["_targets"].detach().cpu().float())
            all_pred_reg.append(_outputs["predict_y_reg"].detach().cpu().float())
            all_pred_exp.append(_outputs["predict_y_exp"].detach().cpu().float())
            all_pred_max.append(_outputs["predict_y_max"].detach().cpu().float())
            if has_rank:
                all_pred_rank.append(_outputs["predict_y_rank"].detach().cpu().float())
            if has_ens:
                all_pred_ens.append(_outputs["predict_y_ens"].detach().cpu().float())

        if not all_paths:
            return

        all_targets_t = torch.cat(all_targets)
        all_pred_reg_t = torch.cat(all_pred_reg)
        all_pred_exp_t = torch.cat(all_pred_exp)
        all_pred_max_t = torch.cat(all_pred_max)
        all_pred_rank_t = torch.cat(all_pred_rank) if has_rank else None
        all_pred_ens_t  = torch.cat(all_pred_ens)  if has_ens  else None

        group_keys = ["targets", "pred_reg", "pred_exp", "pred_max"]
        if has_rank:
            group_keys.append("pred_rank")
        if has_ens:
            group_keys.append("pred_ens")
        video_groups: Dict[str, dict] = defaultdict(
            lambda: {k: [] for k in group_keys}
        )
        for i, path in enumerate(all_paths):
            stem = PurePosixPath(path).stem
            video_id = stem.rsplit("_", 1)[0]
            video_groups[video_id]["targets"].append(all_targets_t[i].item())
            video_groups[video_id]["pred_reg"].append(all_pred_reg_t[i].item())
            video_groups[video_id]["pred_exp"].append(all_pred_exp_t[i].item())
            video_groups[video_id]["pred_max"].append(all_pred_max_t[i].item())
            if has_rank:
                video_groups[video_id]["pred_rank"].append(all_pred_rank_t[i].item())
            if has_ens:
                video_groups[video_id]["pred_ens"].append(all_pred_ens_t[i].item())

        video_mae_reg, video_acc_reg = [], []
        video_mae_exp, video_acc_exp = [], []
        video_mae_max, video_acc_max = [], []
        video_mae_rank, video_acc_rank = [], []
        video_mae_ens, video_acc_ens = [], []
        video_predictions = []

        for vid, data in video_groups.items():
            targets_sorted = sorted(data["targets"])
            gt = targets_sorted[len(targets_sorted) // 2]  # median gt

            pred_reg = sum(data["pred_reg"]) / len(data["pred_reg"])
            pred_exp = sum(data["pred_exp"]) / len(data["pred_exp"])
            pred_max = sum(data["pred_max"]) / len(data["pred_max"])

            video_mae_reg.append(abs(pred_reg - gt))
            video_acc_reg.append(1.0 if round(pred_reg) == gt else 0.0)
            video_mae_exp.append(abs(pred_exp - gt))
            video_acc_exp.append(1.0 if round(pred_exp) == gt else 0.0)
            video_mae_max.append(abs(pred_max - gt))
            video_acc_max.append(1.0 if round(pred_max) == gt else 0.0)

            row = {
                "video_id": vid,
                "gt": int(gt),
                "pred_reg": round(pred_reg, 4),
                "pred_exp": round(pred_exp, 4),
                "pred_max": round(pred_max, 4),
                "n_frames": len(data["targets"]),
            }

            if has_rank:
                pred_rank = sum(data["pred_rank"]) / len(data["pred_rank"])
                video_mae_rank.append(abs(pred_rank - gt))
                video_acc_rank.append(1.0 if round(pred_rank) == gt else 0.0)
                row["pred_rank"] = round(pred_rank, 4)
            if has_ens:
                pred_ens = sum(data["pred_ens"]) / len(data["pred_ens"])
                video_mae_ens.append(abs(pred_ens - gt))
                video_acc_ens.append(1.0 if round(pred_ens) == gt else 0.0)
                row["pred_ens"] = round(pred_ens, 4)

            video_predictions.append(row)

        n_videos = len(video_groups)
        video_stats = {
            # Primary: regression head
            "mae_reg_metric":  sum(video_mae_reg) / n_videos,
            "acc_reg_metric":  sum(video_acc_reg) / n_videos,
            # Ablation: OrdinalCLIP classification
            "mae_exp_metric":  sum(video_mae_exp) / n_videos,
            "acc_exp_metric":  sum(video_acc_exp) / n_videos,
            "mae_max_metric":  sum(video_mae_max) / n_videos,
            "acc_max_metric":  sum(video_acc_max) / n_videos,
            "num_videos": n_videos,
            "epoch": self.current_epoch,
            "output_dir": str(self.output_dir),
            "ckpt_path": str(self.ckpt_path),
        }
        if has_rank:
            video_stats["mae_rank_metric"] = sum(video_mae_rank) / n_videos
            video_stats["acc_rank_metric"] = sum(video_acc_rank) / n_videos
        if has_ens:
            video_stats["mae_ens_metric"]  = sum(video_mae_ens)  / n_videos
            video_stats["acc_ens_metric"]  = sum(video_acc_ens)  / n_videos

        with open(str(self.output_dir / f"{run_type}_video_stats.json"), "a") as f:
            f.write(json.dumps(video_stats) + "\n")

        fieldnames = ["epoch", "ckpt_path", "video_id", "gt",
                      "pred_reg", "pred_exp", "pred_max", "n_frames"]
        if has_rank:
            fieldnames.append("pred_rank")
        if has_ens:
            fieldnames.append("pred_ens")
        csv_path = self.output_dir / f"{run_type}_video_predictions.csv"
        csv_exists = csv_path.exists()
        with open(str(csv_path), "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()
            for row in video_predictions:
                row["epoch"] = self.current_epoch
                row["ckpt_path"] = str(self.ckpt_path)
            writer.writerows(video_predictions)

        log_msg = (
            f"[{run_type}] video-level: {n_videos} videos | "
            f"reg: mae={video_stats['mae_reg_metric']:.4f} acc={video_stats['acc_reg_metric']:.4f} | "
            f"exp: mae={video_stats['mae_exp_metric']:.4f} acc={video_stats['acc_exp_metric']:.4f}"
        )
        if has_rank:
            log_msg += f" | rank: acc={video_stats['acc_rank_metric']:.4f}"
        if has_ens:
            log_msg += f" | ens: acc={video_stats['acc_ens_metric']:.4f}"
        logger.info(log_msg)

    # ================================================================
    #  Metrics
    # ================================================================

    def compute_regression_metrics(
        self, reg_score: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Regression-based MAE and accuracy (primary metric).

        Args:
            reg_score: [B] continuous ŷ ∈ (0, K-1)
            y:         [B] integer ground truth labels

        Returns:
            Dict with mae_reg_metric, acc_reg_metric, predict_y_reg.
        """
        dtype = reg_score.dtype
        y_float = y.type(dtype)
        mae = torch.abs(reg_score - y_float)                      # [B]
        acc = (torch.round(reg_score) == y_float).type(dtype)    # [B]
        return {
            "mae_reg_metric": mae,
            "acc_reg_metric": acc,
            "predict_y_reg":  reg_score,
        }

    def compute_per_example_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, gather_type: str = "exp"
    ) -> Dict[str, torch.Tensor]:
        """OrdinalCLIP classification-based MAE and accuracy (ablation).

        Args:
            logits:      [B, K]
            y:           [B]
            gather_type: "exp" (expectation) or "max" (argmax)
        """
        dtype = logits.dtype
        probs = F.softmax(logits, -1)  # [B, K]

        if gather_type == "exp":
            rank_values = self.rank_output_value_array.type(dtype)
            predict_y = torch.sum(probs * rank_values, dim=-1)  # [B]
        elif gather_type == "max":
            predict_y = torch.argmax(probs, dim=-1).type(dtype)  # [B]
        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")

        y_float = y.type(dtype)
        mae = torch.abs(predict_y - y_float)
        acc = (torch.round(predict_y) == y_float).type(dtype)

        return {
            f"mae_{gather_type}_metric": mae,
            f"acc_{gather_type}_metric": acc,
            f"predict_y_{gather_type}": predict_y,
        }

    @staticmethod
    def _compute_binary_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute AUC via Wilcoxon-Mann-Whitney statistic."""
        n_pos = (labels == 1).sum().item()
        n_neg = (labels == 0).sum().item()
        if n_pos == 0 or n_neg == 0:
            return 0.5
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = labels[sorted_indices].float()
        ranks = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float32)
        pos_rank_sum = ranks[sorted_labels == 1].sum().item()
        auc = 1.0 - (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return max(0.0, min(1.0, auc))

    # ================================================================
    #  Anchor-Based Rank / Ensemble Predictions
    # ================================================================

    def _compute_rank_predictions(
        self, image_features: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Anchor-based rank predictions using concat ranking head.

        For 2-class binary:
          Score = concat_ranking_head([e_x ∥ e_anchor_0]) — P(x > class 0)

        For multi-class K (cumulative link):
          s_k = concat_ranking_head([e_x ∥ e_anchor_k])  for k=0..K-2
          P(class=0)   = 1 - σ(s_0)
          P(class=k)   = σ(s_{k-1}) - σ(s_k)
          P(class=K-1) = σ(s_{K-2})
          Clamp negatives + renormalize.

        Anchors are stored in MLP embedding space (e = SharedMLP(f)).

        Args:
            image_features: [B, D] raw CLIP features
            y:              [B]

        Returns:
            Dict with mae_rank_metric, acc_rank_metric, predict_y_rank, _p_rank.
        """
        dtype = image_features.dtype
        K = self.num_ranks

        # Project test features through SharedMLP
        e_x = self.module.shared_mlp(image_features.type(dtype))  # [B, out]

        if K == 2:
            return self._compute_rank_predictions_binary(e_x, y)

        # Multi-class cumulative link
        cum_probs = []
        for k in range(K - 1):
            anchor_e_k = self._anchors[k].to(e_x.device).type(dtype)  # [out]
            anchor_e_k_expanded = anchor_e_k.unsqueeze(0).expand(e_x.shape[0], -1)  # [B, out]
            concat_feat = torch.cat([e_x, anchor_e_k_expanded], dim=-1)  # [B, 2*out]
            logit_k = self.module.ranking_head(concat_feat)  # [B, 1]
            s_k = torch.sigmoid(logit_k).squeeze(1)          # [B]
            cum_probs.append(s_k)

        cum_probs_t = torch.stack(cum_probs, dim=1)  # [B, K-1]

        ones  = torch.ones(e_x.shape[0], 1, device=e_x.device, dtype=dtype)
        zeros = torch.zeros(e_x.shape[0], 1, device=e_x.device, dtype=dtype)
        boundaries = torch.cat([ones, cum_probs_t, zeros], dim=1)  # [B, K+1]
        p_rank = (boundaries[:, :-1] - boundaries[:, 1:]).clamp(min=0.0)  # [B, K]
        p_rank = p_rank / p_rank.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, K]

        rank_values = self.rank_output_value_array.type(dtype)
        predict_y_rank = torch.sum(p_rank * rank_values, dim=-1)  # [B]

        y_float = y.type(dtype)
        return {
            "mae_rank_metric":  torch.abs(predict_y_rank - y_float),
            "acc_rank_metric":  (torch.round(predict_y_rank) == y_float).type(dtype),
            "predict_y_rank":   predict_y_rank,
            "_p_rank":          p_rank,
        }

    def _compute_rank_predictions_binary(
        self, e_x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Binary (2-class) anchor-based rank predictions."""
        dtype = e_x.dtype
        anchor_e_0 = self._anchors[0].to(e_x.device).type(dtype)  # [out]
        anchor_e_0_exp = anchor_e_0.unsqueeze(0).expand(e_x.shape[0], -1)
        concat_feat = torch.cat([e_x, anchor_e_0_exp], dim=-1)    # [B, 2*out]
        rank_logit = self.module.ranking_head(concat_feat)         # [B, 1]
        rank_score = torch.sigmoid(rank_logit).squeeze(1)          # [B]

        if self._anchor_cfg.get("anchor_mode", "single") == "dual" and 1 in self._anchors:
            anchor_e_1 = self._anchors[1].to(e_x.device).type(dtype)
            anchor_e_1_exp = anchor_e_1.unsqueeze(0).expand(e_x.shape[0], -1)
            concat_feat_1 = torch.cat([e_x, anchor_e_1_exp], dim=-1)
            rank_logit_1 = self.module.ranking_head(concat_feat_1)
            rank_score_1 = torch.sigmoid(rank_logit_1).squeeze(1)
            rank_score = (rank_score + (1.0 - rank_score_1)) / 2.0

        predict_y_rank = rank_score
        p_rank = torch.stack([1.0 - rank_score, rank_score], dim=1)  # [B, 2]

        y_float = y.type(dtype)
        return {
            "mae_rank_metric":  torch.abs(predict_y_rank - y_float),
            "acc_rank_metric":  (torch.round(predict_y_rank) == y_float).type(dtype),
            "predict_y_rank":   predict_y_rank,
            "_p_rank":          p_rank,
        }

    def _compute_ensemble_predictions(
        self,
        p_cls: torch.Tensor,
        p_rank: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Ensemble cls + rank: p_final = α·p_cls + (1-α)·p_rank."""
        alpha = self._anchor_cfg.get("ensemble_alpha", 0.5)
        dtype = p_cls.dtype
        p_ens = alpha * p_cls + (1.0 - alpha) * p_rank  # [B, K]
        rank_values = self.rank_output_value_array.type(dtype)
        predict_y_ens = torch.sum(p_ens * rank_values, dim=-1)  # [B]
        y_float = y.type(dtype)
        return {
            "mae_ens_metric":  torch.abs(predict_y_ens - y_float),
            "acc_ens_metric":  (torch.round(predict_y_ens) == y_float).type(dtype),
            "predict_y_ens":   predict_y_ens,
        }

    # ================================================================
    #  Optimizer & Scheduler
    # ================================================================

    def configure_optimizers(self):
        return self._build_optimizer_and_scheduler(**self._optimizer_and_scheduler_cfg)

    def _build_optimizer_and_scheduler(
        self,
        param_dict_cfg: dict,
        optimizer_cfg: dict,
        lr_scheduler_cfg: dict,
    ):
        param_dict_ls = self.build_param_dict(**param_dict_cfg)
        optim = build_optimizer(model=param_dict_ls, **optimizer_cfg)
        sched = build_lr_scheduler(optimizer=optim, **lr_scheduler_cfg)
        return [optim], [sched]

    def build_param_dict(
        self,
        lr_siamese_heads: float = 1e-4,
        lr_backbone: float = 0.0,
    ) -> List[dict]:
        """Parameter groups for optimizer.

        Args:
            lr_siamese_heads: LR for SharedMLP + RegressionHead + ConcatRankingHead.
            lr_backbone:      LR for backbone. 0 = frozen.
        """
        param_dict_ls = []

        # Siamese heads (always trained)
        siamese_params = (
            list(self.module.shared_mlp.parameters())
            + list(self.module.regression_head.parameters())
            + list(self.module.ranking_head.parameters())
        )
        if lr_siamese_heads > 0:
            param_dict_ls.append({
                "params": siamese_params,
                "lr": lr_siamese_heads,
                "init_lr": lr_siamese_heads,
                "name": "siamese_heads",
            })
        else:
            self._custom_logger.warning("lr_siamese_heads=0, heads will NOT be trained!")

        # Backbone (optional fine-tune)
        if lr_backbone > 0:
            if self._freeze_backbone:
                self._custom_logger.warning(
                    "freeze_backbone=True but lr_backbone>0: unfreezing backbone."
                )
            self._custom_logger.info(f"Unfreezing backbone with lr={lr_backbone}")
            for param in self.module.backbone.parameters():
                param.requires_grad = True
            param_dict_ls.append({
                "params": list(self.module.backbone.parameters()),
                "lr": lr_backbone,
                "init_lr": lr_backbone,
                "name": "backbone",
            })
        else:
            self._custom_logger.info("Backbone frozen (lr_backbone=0)")

        if not param_dict_ls:
            raise ValueError(
                "No trainable parameters! lr_siamese_heads and lr_backbone are both 0."
            )

        return param_dict_ls

    # ================================================================
    #  Lifecycle hooks
    # ================================================================

    def on_train_epoch_start(self) -> None:
        param_group_lrs = {
            pg.get("name", "<unnamed>"): (pg["lr"], len(list(pg["params"])))
            for pg in self.optimizers().param_groups
        }
        logger.info(
            f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: "
            f"{param_group_lrs}"
        )

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    def on_fit_end(self) -> None:
        """Compute and save anchors after training completes."""
        if not self._anchor_cfg.get("enabled", False):
            return
        logger.info("Computing class anchors from training data (MLP embedding space)...")
        self._anchors = self._compute_anchors()
        if self._anchor_cfg.get("save_anchors", True):
            self._save_anchors(self._anchors)

    def on_test_start(self) -> None:
        """Load or compute anchors before test begins."""
        if not self._anchor_cfg.get("enabled", False):
            return
        if self._anchors is not None:
            return

        load_path = self._anchor_cfg.get("load_anchors_path")
        if load_path is None:
            load_path = str(self.output_dir / "anchors.pt")

        if Path(load_path).exists():
            self._anchors = self._load_anchors(load_path)
        else:
            if self.trainer.datamodule is None:
                raise RuntimeError(
                    f"No saved anchors at '{load_path}' and trainer.datamodule is None."
                )
            logger.info("No saved anchors found, computing from training data...")
            self._anchors = self._compute_anchors()
            if self._anchor_cfg.get("save_anchors", True):
                self._save_anchors(self._anchors)

    # ================================================================
    #  Anchor Computation (in MLP embedding space)
    # ================================================================

    @torch.no_grad()
    def _compute_anchors(self) -> Dict[int, torch.Tensor]:
        """Compute per-class feature centroids in SharedMLP embedding space.

        Iterates training set with eval transforms, extracts CLIP features,
        passes through SharedMLP, then averages per class.
        Centroids are L2-normalized before storage.
        """
        anchor_loader = self.trainer.datamodule.anchor_dataloader()
        feat_accum: Dict[int, List[torch.Tensor]] = defaultdict(list)

        was_training = self.training
        self.eval()

        try:
            for batch in anchor_loader:
                images = batch[0].to(self.device)
                labels = batch[1]
                _, image_features, _ = self.module.forward_single(images)
                # image_features: [B, D] raw CLIP features
                if image_features is None:
                    raise RuntimeError(
                        "Backbone does not produce image_features (e.g. Baseline). "
                        "Anchor inference requires OrdinalCLIP backbone."
                    )
                # Project through SharedMLP to get anchor embedding
                e = self.module.shared_mlp(image_features.float())  # [B, out]
                e = e.cpu()
                for i in range(labels.size(0)):
                    feat_accum[labels[i].item()].append(e[i])
        finally:
            if was_training:
                self.train()

        expected_labels = set(range(self.num_ranks))
        actual_labels = set(feat_accum.keys())
        if not expected_labels.issubset(actual_labels):
            missing = expected_labels - actual_labels
            raise ValueError(
                f"Anchor computation failed: classes {missing} not found in training data."
            )

        anchors: Dict[int, torch.Tensor] = {}
        for label in sorted(feat_accum.keys()):
            stacked = torch.stack(feat_accum[label])   # [N_label, out]
            centroid = stacked.mean(dim=0)              # [out]
            centroid = F.normalize(centroid, dim=0)     # L2-normalize
            anchors[label] = centroid
            logger.info(
                f"Anchor class {label}: {len(feat_accum[label])} samples, "
                f"centroid norm={centroid.norm().item():.4f}"
            )

        return anchors

    def _save_anchors(self, anchors: Dict[int, torch.Tensor]) -> None:
        path = self.output_dir / "anchors.pt"
        torch.save(anchors, str(path))
        logger.info(f"Saved {len(anchors)} class anchors (MLP space) to {path}")

    def _load_anchors(self, path: str) -> Dict[int, torch.Tensor]:
        # Note: anchors.pt contains only {int: Tensor} — safe to unpickle.
        # TODO: add weights_only=True when upgrading to PyTorch >= 1.13.
        anchors = torch.load(path, map_location="cpu")
        logger.info(f"Loaded {len(anchors)} class anchors from {path}")
        return anchors

    # ================================================================
    #  Model IO
    # ================================================================

    def _load_backbone_weights(self, backbone_ckpt_path: Optional[str] = None) -> None:
        """Load Stage 1 backbone weights from a PL checkpoint.

        Strips the 'module.' prefix from Runner's state dict keys and loads
        into self.module.backbone.
        """
        if backbone_ckpt_path is None:
            self._custom_logger.info("No backbone_ckpt_path provided, skip loading.")
            return

        self._custom_logger.info(f"Loading backbone from: {backbone_ckpt_path}")
        map_location = "cpu" if not torch.cuda.is_available() else None
        # Trust assumption: backbone checkpoints are locally produced by Stage 1.
        # TODO: add weights_only=True when upgrading to PyTorch >= 1.13.
        ckpt = torch.load(backbone_ckpt_path, map_location=map_location)

        state_dict = ckpt.get("state_dict", ckpt)

        backbone_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                backbone_state[k[7:]] = v
            else:
                backbone_state[k] = v

        model_dict = self.module.backbone.state_dict()
        matched = {
            k: v for k, v in backbone_state.items()
            if k in model_dict and model_dict[k].size() == v.size()
        }
        model_dict.update(matched)
        self.module.backbone.load_state_dict(model_dict)

        self._custom_logger.info(
            f"Loaded {len(matched)}/{len(model_dict)} backbone weights "
            f"from {backbone_ckpt_path}"
        )
        if len(matched) == 0:
            self._custom_logger.warning(
                "WARNING: 0 backbone weights matched! Check checkpoint compatibility."
            )

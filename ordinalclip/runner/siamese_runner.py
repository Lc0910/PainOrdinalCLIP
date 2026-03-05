"""Siamese Runner for Stage 2 pairwise ranking training.

Key design decisions (from Codex review):
  P0-1: Joint loss as default — CE_a + CE_b + λ * BCE_rank
  P0-2: Cross-subject pair sampling is enforced in siamese_data.py
  P1-1: Supports frozen backbone (E/F/H) and unfrozen backbone (G)
  P1-2: Head type (linear/mlp) configured via ranking_head_cfg
  P1-3: Pairwise accuracy + AUC reported during training

Training:
  - Pairwise input → ranking BCE loss + CE loss for each image
  - Metrics: pairwise accuracy, pairwise AUC (epoch-level)

Val/Test:
  - Single-image classification via forward_single()
  - Frame-level MAE/acc + video-level aggregation (same as Runner)
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
from .utils import load_pretrained_weights

logger = get_logger(__name__)


class SiameseRunner(pl.LightningModule):
    """PyTorch Lightning module for Siamese ranking Stage 2 training.

    Wraps a SiameseOrdinalCLIP model (frozen backbone + ranking head).
    """

    def __init__(
        self,
        backbone_cfg: dict,
        ranking_head_cfg: dict,
        output_dir: str,
        optimizer_and_scheduler_cfg: dict,
        load_weights_cfg: dict,
        seed: int,
        loss_weights: Optional[Dict] = None,
        freeze_backbone: bool = True,
        ckpt_path: str = "",
    ) -> None:
        super().__init__()

        if loss_weights is None:
            loss_weights = {"ranking_loss": 1.0, "ce_loss_a": 0.5, "ce_loss_b": 0.5}

        # --- 1. Build backbone from config (OrdinalCLIP / Baseline) ---
        backbone = MODELS.build(backbone_cfg)

        # --- 2. Inject embed_dims into ranking_head_cfg ---
        ranking_head_cfg = dict(ranking_head_cfg)  # copy to avoid mutation
        ranking_head_cfg["embed_dims"] = backbone.embed_dims

        # --- 3. Build Siamese model ---
        self.module = SiameseOrdinalCLIP(
            backbone=backbone,
            ranking_head_cfg=ranking_head_cfg,
            freeze_backbone=freeze_backbone,
        )

        # --- Output & logger (must be initialized before _load_backbone_weights) ---
        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path

        # --- 4. Load backbone weights from Stage 1 checkpoint ---
        self._load_backbone_weights(**load_weights_cfg)

        # --- Loss functions ---
        self.ranking_loss_func = nn.BCEWithLogitsLoss()
        self.ce_loss_func = nn.CrossEntropyLoss()
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
        self._train_ranking_probs: List[torch.Tensor] = []
        self._train_ranking_labels: List[torch.Tensor] = []

    # ================================================================
    #  Forward
    # ================================================================

    def forward(self, images_a: torch.Tensor, images_b: torch.Tensor):
        """Pairwise forward for training."""
        return self.module(images_a, images_b)

    def forward_single(self, images: torch.Tensor):
        """Single-image forward for val/test classification."""
        return self.module.forward_single(images)

    # ================================================================
    #  Training
    # ================================================================

    def training_step(self, batch, batch_idx):
        img_a, img_b, pair_label, rank_a, rank_b = batch
        # img_a/img_b: [B, 3, H, W], pair_label: [B], rank_a/rank_b: [B]

        ranking_logits, logits_a, logits_b = self.module(img_a, img_b)
        # ranking_logits: [B, 1], logits_a: [B, num_ranks], logits_b: [B, num_ranks]

        # --- Losses ---
        pair_target = pair_label.float().unsqueeze(1)  # [B, 1]
        ranking_loss = self.ranking_loss_func(ranking_logits, pair_target)
        ce_loss_a = self.ce_loss_func(logits_a, rank_a)
        ce_loss_b = self.ce_loss_func(logits_b, rank_b)

        loss = (
            self.loss_weights.get("ranking_loss", 1.0) * ranking_loss
            + self.loss_weights.get("ce_loss_a", 0.5) * ce_loss_a
            + self.loss_weights.get("ce_loss_b", 0.5) * ce_loss_b
        )

        # --- Pairwise accuracy (per-batch) ---
        with torch.no_grad():
            pairwise_pred = (ranking_logits.squeeze(1) > 0).long()  # [B]
            pairwise_acc = (pairwise_pred == pair_label).float().mean()

            # Accumulate for epoch-level AUC
            probs = torch.sigmoid(ranking_logits.squeeze(1)).detach()  # [B]
            self._train_ranking_probs.append(probs.cpu())
            self._train_ranking_labels.append(pair_label.cpu())

        # --- Logging ---
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_ranking_loss", ranking_loss, on_step=True, on_epoch=True)
        self.log("train_ce_loss_a", ce_loss_a, on_step=True, on_epoch=True)
        self.log("train_ce_loss_b", ce_loss_b, on_step=True, on_epoch=True)
        self.log("train_pairwise_acc", pairwise_acc, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        """Compute epoch-level pairwise AUC (P1-3)."""
        if self._train_ranking_probs:
            all_probs = torch.cat(self._train_ranking_probs)  # [N]
            all_labels = torch.cat(self._train_ranking_labels)  # [N]
            auc = self._compute_binary_auc(all_probs, all_labels)
            self.log("train_pairwise_auc", auc, on_epoch=True, prog_bar=True)

        # Reset accumulators
        self._train_ranking_probs.clear()
        self._train_ranking_labels.clear()

    # ================================================================
    #  Validation / Test  (single-image classification, same as Runner)
    # ================================================================

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch)

    def _eval_step(self, batch):
        """Single-image classification evaluation (reuses Runner logic)."""
        x, y = batch[0], batch[1]  # [B, 3, H, W], [B]
        logits, *_ = self.module.forward_single(x)  # [B, num_ranks]

        ce_loss = self.ce_loss_func(logits, y)
        metrics_exp = self.compute_per_example_metrics(logits, y, "exp")
        metrics_max = self.compute_per_example_metrics(logits, y, "max")

        outputs = {"loss": ce_loss, "ce_loss": ce_loss, **metrics_exp, **metrics_max}
        if len(batch) > 2:
            outputs["_paths"] = batch[2]
            outputs["_targets"] = batch[1]
        return outputs

    def validation_epoch_end(self, outputs) -> None:
        self._eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self._eval_epoch_end(outputs, "test")

    # ================================================================
    #  Epoch-end aggregation (adapted from Runner)
    # ================================================================

    loggings_suffix = {"metric", "loss"}

    def _valid_key(self, key: str) -> bool:
        return any(key.endswith(suffix) for suffix in self.loggings_suffix)

    def _eval_epoch_end(self, outputs, run_type: str) -> None:
        """Frame-level stats + video-level aggregation."""
        # --- Frame-level stats ---
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
            self.log(
                f"{run_type}_{k}", stats[k],
                on_step=False, on_epoch=True, prog_bar=False, logger=True,
            )

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

        # --- Video-level aggregation ---
        self._video_level_aggregation(outputs, run_type)

    def _video_level_aggregation(self, outputs, run_type: str) -> None:
        """Aggregate frame predictions to video-level results.

        video_id derived by stripping trailing frame index from filename:
            images/071309_w_21-BL1-081_27.jpg -> 071309_w_21-BL1-081
        """
        all_paths: List[str] = []
        all_targets: List[torch.Tensor] = []
        all_pred_exp: List[torch.Tensor] = []
        all_pred_max: List[torch.Tensor] = []

        for _outputs in outputs:
            if "_paths" not in _outputs:
                return
            all_paths.extend(_outputs["_paths"])
            all_targets.append(_outputs["_targets"].detach().cpu().float())
            all_pred_exp.append(_outputs["predict_y_exp"].detach().cpu().float())
            all_pred_max.append(_outputs["predict_y_max"].detach().cpu().float())

        if not all_paths:
            return

        all_targets_t = torch.cat(all_targets)  # [N]
        all_pred_exp_t = torch.cat(all_pred_exp)  # [N]
        all_pred_max_t = torch.cat(all_pred_max)  # [N]

        # Group by video_id
        video_groups: Dict[str, dict] = defaultdict(
            lambda: {"targets": [], "pred_exp": [], "pred_max": []}
        )
        for i, path in enumerate(all_paths):
            stem = PurePosixPath(path).stem
            video_id = stem.rsplit("_", 1)[0]
            video_groups[video_id]["targets"].append(all_targets_t[i].item())
            video_groups[video_id]["pred_exp"].append(all_pred_exp_t[i].item())
            video_groups[video_id]["pred_max"].append(all_pred_max_t[i].item())

        # Compute video-level metrics
        video_mae_exp, video_mae_max = [], []
        video_acc_exp, video_acc_max = [], []
        video_predictions = []

        for vid, data in video_groups.items():
            targets_sorted = sorted(data["targets"])
            gt = targets_sorted[len(targets_sorted) // 2]  # median gt
            pred_exp = sum(data["pred_exp"]) / len(data["pred_exp"])
            pred_max = sum(data["pred_max"]) / len(data["pred_max"])

            video_mae_exp.append(abs(pred_exp - gt))
            video_mae_max.append(abs(pred_max - gt))
            video_acc_exp.append(1.0 if round(pred_exp) == gt else 0.0)
            video_acc_max.append(1.0 if round(pred_max) == gt else 0.0)

            video_predictions.append({
                "video_id": vid,
                "gt": int(gt),
                "pred_exp": round(pred_exp, 4),
                "pred_max": round(pred_max, 4),
                "n_frames": len(data["targets"]),
            })

        n_videos = len(video_groups)
        video_stats = {
            "mae_exp_metric": sum(video_mae_exp) / n_videos,
            "mae_max_metric": sum(video_mae_max) / n_videos,
            "acc_exp_metric": sum(video_acc_exp) / n_videos,
            "acc_max_metric": sum(video_acc_max) / n_videos,
            "num_videos": n_videos,
            "epoch": self.current_epoch,
            "output_dir": str(self.output_dir),
            "ckpt_path": str(self.ckpt_path),
        }
        with open(str(self.output_dir / f"{run_type}_video_stats.json"), "a") as f:
            f.write(json.dumps(video_stats) + "\n")

        # Per-video predictions CSV
        csv_path = self.output_dir / f"{run_type}_video_predictions.csv"
        csv_exists = csv_path.exists()
        with open(str(csv_path), "a", newline="") as f:
            fieldnames = ["epoch", "ckpt_path", "video_id", "gt", "pred_exp", "pred_max", "n_frames"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()
            for row in video_predictions:
                row["epoch"] = self.current_epoch
                row["ckpt_path"] = str(self.ckpt_path)
            writer.writerows(video_predictions)

        logger.info(
            f"[{run_type}] video-level: {n_videos} videos, "
            f"mae_exp={video_stats['mae_exp_metric']:.4f}, "
            f"mae_max={video_stats['mae_max_metric']:.4f}, "
            f"acc_exp={video_stats['acc_exp_metric']:.4f}, "
            f"acc_max={video_stats['acc_max_metric']:.4f}"
        )

    # ================================================================
    #  Metrics
    # ================================================================

    def compute_per_example_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, gather_type: str = "exp"
    ) -> Dict[str, torch.Tensor]:
        """Compute MAE and accuracy per example (same as Runner)."""
        dtype = logits.dtype
        probs = F.softmax(logits, -1)  # [B, num_ranks]

        if gather_type == "exp":
            rank_values = self.rank_output_value_array.type(dtype)  # [num_ranks]
            predict_y = torch.sum(probs * rank_values, dim=-1)  # [B]
        elif gather_type == "max":
            predict_y = torch.argmax(probs, dim=-1).type(dtype)  # [B]
        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")

        y_float = y.type(dtype)
        mae = torch.abs(predict_y - y_float)  # [B]
        acc = (torch.round(predict_y) == y_float).type(dtype)  # [B]

        return {
            f"mae_{gather_type}_metric": mae,
            f"acc_{gather_type}_metric": acc,
            f"predict_y_{gather_type}": predict_y,
        }

    @staticmethod
    def _compute_binary_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute AUC for binary ranking via Wilcoxon-Mann-Whitney statistic.

        Args:
            probs:  [N] predicted probabilities (higher → more likely positive).
            labels: [N] binary labels (0 or 1).

        Returns:
            AUC score in [0, 1].
        """
        n_pos = (labels == 1).sum().item()
        n_neg = (labels == 0).sum().item()
        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Sort by descending probability
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_labels = labels[sorted_indices].float()

        # Count: for each positive, how many negatives are ranked below it
        # Equivalent to: sum of ranks of positives - n_pos*(n_pos+1)/2
        ranks = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float32)
        pos_rank_sum = ranks[sorted_labels == 1].sum().item()
        auc = 1.0 - (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return max(0.0, min(1.0, auc))

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
        lr_ranking_head: float = 1e-3,
        lr_backbone: float = 0.0,
    ) -> List[dict]:
        """Build parameter groups for optimizer.

        Args:
            lr_ranking_head: Learning rate for ranking head.
            lr_backbone: Learning rate for backbone. 0 = frozen.
        """
        param_dict_ls = []

        # Ranking head (always trained)
        if lr_ranking_head > 0:
            param_dict_ls.append({
                "params": list(self.module.ranking_head.parameters()),
                "lr": lr_ranking_head,
                "init_lr": lr_ranking_head,
                "name": "ranking_head",
            })
        else:
            self._custom_logger.warning("lr_ranking_head=0, ranking head will NOT be trained!")

        # Backbone (optional fine-tune)
        if lr_backbone > 0:
            if self._freeze_backbone:
                self._custom_logger.warning(
                    "freeze_backbone=True but lr_backbone>0: unfreezing backbone for training."
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
                "No trainable parameters! Both lr_ranking_head and lr_backbone are 0. "
                "Set at least lr_ranking_head > 0."
            )

        return param_dict_ls

    # ================================================================
    #  Lifecycle hooks
    # ================================================================

    def on_train_epoch_start(self) -> None:
        param_group_lrs = {
            pg["name"]: (pg["lr"], len(list(pg["params"])))
            for pg in self.optimizers().param_groups
        }
        logger.info(
            f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: "
            f"{param_group_lrs}"
        )

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    # ================================================================
    #  Model IO
    # ================================================================

    def _load_backbone_weights(self, backbone_ckpt_path: Optional[str] = None) -> None:
        """Load Stage 1 backbone weights from a PL checkpoint.

        The checkpoint is saved by Runner and has state keys like:
            module.image_encoder.*, module.prompt_learner.*, ...
        We strip the 'module.' prefix and load into self.module.backbone.
        """
        if backbone_ckpt_path is None:
            self._custom_logger.info("No backbone_ckpt_path provided, skip loading.")
            return

        self._custom_logger.info(f"Loading backbone from: {backbone_ckpt_path}")
        map_location = "cpu" if not torch.cuda.is_available() else None
        ckpt = torch.load(backbone_ckpt_path, map_location=map_location)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Strip 'module.' prefix (Runner saves as self.module.*)
        backbone_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                backbone_state[k[7:]] = v
            else:
                backbone_state[k] = v

        # Match and load
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

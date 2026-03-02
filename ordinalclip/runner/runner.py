import csv
import json
from collections import defaultdict
from multiprocessing.sharedctypes import Value
from pathlib import Path, PurePosixPath

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ordinalclip.models import MODELS
from ordinalclip.models.ordinalclip import OrdinalCLIP
from ordinalclip.utils.logging import get_logger

from .optim import build_lr_scheduler, build_optimizer, build_staged_lr_param_groups
from .utils import freeze_param, load_pretrained_weights

logger = get_logger(__name__)


class Runner(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        output_dir: str,
        optimizer_and_scheduler_cfg,
        load_weights_cfg,
        seed: int,
        loss_weights=dict(
            ce_loss=1.0,
            kl_loss=1.0,
        ),
        ckpt_path="",
    ) -> None:
        super().__init__()
        self.module = MODELS.build(model_cfg)

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.kl_loss_func = nn.KLDivLoss(reduction="sum")
        self.loss_weights = loss_weights
        self.num_ranks = self.module.num_ranks
        self.register_buffer("rank_output_value_array", torch.arange(0, self.num_ranks).float(), persistent=False)
        self.output_dir = Path(output_dir)
        self._custom_logger = get_logger(__name__)

        self.load_weights(**load_weights_cfg)
        self._optimizer_and_scheduler_cfg = optimizer_and_scheduler_cfg
        self.seed = seed
        self.ckpt_path = ckpt_path

    # Model Forward
    def forward(self, images):
        return self.module(images)

    def forward_text_only(self):
        return self.forward_text_only()

    # Running Steps
    def run_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]  # [B, 3, H, W], [B]
        logits, *_ = self.module(x)  # logits: [B, num_ranks]

        losses = self.compute_losses(logits, y)
        loss = sum([weight * losses[k] for k, weight in self.loss_weights.items()])

        metrics_exp = self.compute_per_example_metrics(logits, y, "exp")
        metrics_max = self.compute_per_example_metrics(logits, y, "max")
        return {"loss": loss, **losses, **metrics_exp, **metrics_max}

    def training_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)

        self.logging(outputs, "train", on_step=True, on_epoch=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)
        if len(batch) > 2:
            outputs["_paths"] = batch[2]
            outputs["_targets"] = batch[1]
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)
        if len(batch) > 2:
            outputs["_paths"] = batch[2]
            outputs["_targets"] = batch[1]
        return outputs

    # Epoch Eval
    def eval_epoch_end(self, outputs, run_type):
        """Frame-level stats + video-level aggregation.

        Args:
            outputs: list of dicts from val/test steps
            run_type: "val" or "test"
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        # --- Frame-level stats (original logic) ---
        stats = defaultdict(list)
        for _outputs in outputs:
            for k, v in _outputs.items():
                if self._valid_key(k):
                    stats[k].append(v)
        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

        # --- Video-level aggregation ---
        self._video_level_aggregation(outputs, run_type)

    def _video_level_aggregation(self, outputs, run_type):
        """Aggregate frame predictions to video-level results.

        video_id is derived by stripping the trailing frame index:
        e.g. "images/071309_w_21-BL1-081_27.jpg" -> video_id "071309_w_21-BL1-081"
        """
        all_paths = []
        all_targets = []
        all_pred_exp = []
        all_pred_max = []
        for _outputs in outputs:
            if "_paths" not in _outputs:
                return  # paths not available, skip video aggregation
            all_paths.extend(_outputs["_paths"])
            all_targets.append(_outputs["_targets"].detach().cpu().float())
            all_pred_exp.append(_outputs["predict_y_exp"].detach().cpu().float())
            all_pred_max.append(_outputs["predict_y_max"].detach().cpu().float())

        if not all_paths:
            return

        all_targets = torch.cat(all_targets)  # [N]
        all_pred_exp = torch.cat(all_pred_exp)  # [N]
        all_pred_max = torch.cat(all_pred_max)  # [N]

        # Parse video_id from path: remove extension, split by last "_" to drop frame index
        video_groups = defaultdict(lambda: {"targets": [], "pred_exp": [], "pred_max": []})
        for i, path in enumerate(all_paths):
            stem = PurePosixPath(path).stem  # e.g. "071309_w_21-BL1-081_27"
            video_id = stem.rsplit("_", 1)[0]  # e.g. "071309_w_21-BL1-081"
            video_groups[video_id]["targets"].append(all_targets[i].item())
            video_groups[video_id]["pred_exp"].append(all_pred_exp[i].item())
            video_groups[video_id]["pred_max"].append(all_pred_max[i].item())

        # Compute video-level metrics
        video_mae_exp, video_mae_max = [], []
        video_acc_exp, video_acc_max = [], []
        video_predictions = []

        for vid, data in video_groups.items():
            targets_sorted = sorted(data["targets"])
            gt = targets_sorted[len(targets_sorted) // 2]  # median gt
            pred_exp = sum(data["pred_exp"]) / len(data["pred_exp"])  # mean pred
            pred_max = sum(data["pred_max"]) / len(data["pred_max"])  # mean pred

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

        # Write per-video predictions CSV.
        # Append instead of overwrite to preserve history across epochs/ckpts.
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
            f"mae_exp={video_stats['mae_exp_metric']:.4f}, mae_max={video_stats['mae_max_metric']:.4f}, "
            f"acc_exp={video_stats['acc_exp_metric']:.4f}, acc_max={video_stats['acc_max_metric']:.4f}"
        )

    def validation_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs) -> None:
        self.eval_epoch_end(outputs, "test")

    def on_train_epoch_start(self) -> None:
        param_group_lrs = {pg["name"]: (pg["lr"], len(list(pg["params"]))) for pg in self.optimizers().param_groups}
        logger.info(f"check optimizer `param_groups` lr @ epoch {self.current_epoch}: {param_group_lrs}")

    def on_fit_start(self) -> None:
        pl.seed_everything(self.seed, workers=True)

    # Logging Utils
    loggings_suffix = {"metric", "loss"}

    def _valid_key(self, key: str):
        for suffix in self.loggings_suffix:
            if key.endswith(suffix):
                return True
        else:
            return False

    def logging(self, outputs: dict, run_type: str, on_step=True, on_epoch=True):
        for k, v in outputs.items():
            if self._valid_key(k):
                self.log(f"{run_type}_{k}", v.mean(), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)

    # Loss & Metrics
    def compute_losses(self, logits, y):
        losses = {}
        losses["ce_loss"] = self.ce_loss_func(logits, y)
        losses["kl_loss"] = self.compute_kl_loss(logits, y)

        return losses

    def compute_kl_loss(self, logits, y):
        y_t = F.one_hot(y, self.num_ranks).t()
        y_t_row_ind = y_t.sum(-1) > 0
        num_slots = y_t_row_ind.sum()
        y_t_reduction = (y_t * 10.0).softmax(-1)
        y_t_reduction[y_t_row_ind <= 0] = 0

        logits_t = logits.t()
        kl_loss = self.kl_loss_func(F.log_softmax(logits_t, dim=-1), y_t_reduction) / num_slots
        return kl_loss

    def compute_per_example_metrics(self, logits, y, gather_type="exp"):
        dtype = logits.dtype
        probs = F.softmax(logits, -1)

        if gather_type == "exp":
            rank_output_value_array = self.rank_output_value_array.type(dtype)
            predict_y = torch.sum(probs * rank_output_value_array, dim=-1)
        elif gather_type == "max":
            predict_y = torch.argmax(probs, dim=-1).type(dtype)
        else:
            raise ValueError(f"Invalid gather_type: {gather_type}")

        y = y.type(dtype)
        mae = torch.abs(predict_y - y)
        acc = (torch.round(predict_y) == y).type(logits.dtype)

        return {f"mae_{gather_type}_metric": mae, f"acc_{gather_type}_metric": acc, f"predict_y_{gather_type}": predict_y}

    # Optimizer & Scheduler
    def configure_optimizers(self):
        return self.build_optmizer_and_scheduler(**self._optimizer_and_scheduler_cfg)

    def build_optmizer_and_scheduler(
        self,
        param_dict_cfg=None,
        optimizer_cfg=None,
        lr_scheduler_cfg=None,
    ):
        param_dict_ls = self.build_param_dict(**param_dict_cfg)

        optim = build_optimizer(
            model=param_dict_ls,
            **optimizer_cfg,
        )
        sched = build_lr_scheduler(optimizer=optim, **lr_scheduler_cfg)
        return [optim], [sched]

    # Model IO
    def load_weights(
        self,
        init_model_weights=None,
        init_prompt_learner_weights=None,
        init_image_encoder_weights=None,
        init_text_encoder_weights=None,
    ):
        if init_model_weights is not None:
            self._custom_logger.info("init_model_weights")
            load_pretrained_weights(self.module, init_model_weights)
            return

        if init_prompt_learner_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.prompt_learner, init_prompt_learner_weights)
        if init_image_encoder_weights is not None:
            self._custom_logger.info("init_image_encoder_weights")
            load_pretrained_weights(self.module.image_encoder, init_image_encoder_weights)
        if init_text_encoder_weights is not None:
            self._custom_logger.info("init_prompt_learner_weights")
            load_pretrained_weights(self.module.text_encoder, init_text_encoder_weights)
        return

    def build_param_dict(
        self,
        lr_prompt_learner_context,
        lr_prompt_learner_ranks,
        lr_image_encoder,
        lr_text_encoder,
        lr_logit_scale,
        staged_lr_image_encoder,
    ):
        param_dict_ls = []
        if lr_prompt_learner_context > 0 and self.module.prompt_learner is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.context_embeds,
                    "lr": lr_prompt_learner_context,
                    "init_lr": lr_prompt_learner_context,
                    "name": "lr_prompt_learner_context",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.prompt_learner.context_embeds)")
            try:
                freeze_param(self.module.prompt_learner.context_embeds)
            except AttributeError:
                pass

        if lr_prompt_learner_ranks > 0 and self.module.prompt_learner is not None:
            param_dict_ls.append(
                {
                    "params": self.module.prompt_learner.rank_embeds,
                    "lr": lr_prompt_learner_ranks,
                    "init_lr": lr_prompt_learner_ranks,
                    "name": "lr_prompt_learner_ranks",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.prompt_learner.rank_embeds)")
            try:
                freeze_param(self.module.prompt_learner.rank_embeds)
            except AttributeError:
                pass

        if lr_image_encoder > 0 and self.module.image_encoder is not None:
            if staged_lr_image_encoder is not None:
                self._custom_logger.info("staged_lr_image_encoder activated")
                image_encoder_param_groups = build_staged_lr_param_groups(
                    model=self.module.image_encoder,
                    lr=lr_image_encoder,
                    **staged_lr_image_encoder,
                )
                param_dict_ls.extend(image_encoder_param_groups)
            else:
                param_dict_ls.append(
                    {
                        "params": self.module.image_encoder.parameters(),
                        "lr": lr_image_encoder,
                        "init_lr": lr_image_encoder,
                        "name": "image_encoder",
                    }
                )

        else:
            self._custom_logger.info("freeze_param(self.model.image_encoder)")
            freeze_param(self.module.image_encoder)

        if lr_text_encoder > 0 and self.module.text_encoder is not None:
            param_dict_ls.append(
                {
                    "params": self.module.text_encoder.parameters(),
                    "lr": lr_text_encoder,
                    "init_lr": lr_text_encoder,
                    "name": "text_encoder",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.text_encoder)")
            freeze_param(self.module.text_encoder)

        if lr_logit_scale > 0 and self.module.logit_scale is not None:
            param_dict_ls.append(
                {
                    "params": self.module.logit_scale,
                    "lr": lr_logit_scale,
                    "init_lr": lr_logit_scale,
                    "name": "logit_scale",
                }
            )
        else:
            self._custom_logger.info("freeze_param(self.model.logit_scale)")
            freeze_param(self.module.logit_scale)
        return param_dict_ls

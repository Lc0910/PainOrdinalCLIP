"""FER Baseline model — decoupled from CLIP, uses FER-pretrained backbone.

Replaces CLIP's image encoder with a facial expression recognition (FER)
pretrained backbone (DAN, ViT-FER, etc.). No text encoder, no prompt learner.

The model follows the same interface as Baseline so that Runner can use it
without modification:
    forward(images) -> (logits [B, K], image_features [B, D], None)

Usage:
    Configure via YAML:
        runner_cfg:
          model_cfg:
            type: FERBaseline
            image_encoder_name: dan_resnet18
            fer_weights_path: .cache/fer/dan_affecnet7.pth
            num_ranks: 5
            embed_dims: 512
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ordinalclip.utils import get_logger

from .builder import MODELS
from . import fer_encoders

logger = get_logger(__name__)


def _build_fer_encoder(
    encoder_name: str,
    num_classes: int,
    pretrained_path: str | None = None,
) -> nn.Module:
    """Instantiate a FER encoder by name.

    Looks up the encoder factory in fer_encoders module. Supported names:
        - dan_resnet18
        - vit_fer_base
    """
    factory = getattr(fer_encoders, encoder_name, None)
    if factory is None:
        raise ValueError(
            f"Unknown FER encoder: {encoder_name}. "
            f"Available: {[n for n in dir(fer_encoders) if not n.startswith('_')]}"
        )

    kwargs = {"num_classes": num_classes}
    if pretrained_path:
        kwargs["pretrained_path"] = pretrained_path

    return factory(**kwargs)


@MODELS.register_module()
class FERBaseline(nn.Module):
    """FER-based baseline for ordinal pain regression.

    Architecture:
        Image -> FER_Encoder -> feat [B, D]
                                     |
                              Linear(D, K) -> logits [B, K]

    Compatible with Runner: exposes text_encoder=None, prompt_learner=None,
    logit_scale=None so that build_param_dict handles it correctly.

    Args:
        image_encoder_name: FER encoder factory name (e.g. 'dan_resnet18').
        fer_weights_path: Path to pretrained FER checkpoint, or None for
            ImageNet-only backbone.
        num_ranks: Number of ordinal ranks (classes).
        embed_dims: Feature dimension (output of FER encoder).
        prompt_learner_cfg: Ignored; kept for config compatibility with
            Baseline. Only ``num_ranks`` is read from it if ``num_ranks``
            is not provided directly.
    """

    def __init__(
        self,
        image_encoder_name: str,
        fer_weights_path: str | None = None,
        num_ranks: int | None = None,
        embed_dims: int = 512,
        prompt_learner_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            logger.info(f"FERBaseline: irrelevant kwargs: {kwargs}")

        # Resolve num_ranks from prompt_learner_cfg if not given directly
        if num_ranks is None:
            if prompt_learner_cfg is not None and "num_ranks" in prompt_learner_cfg:
                num_ranks = prompt_learner_cfg["num_ranks"]
            else:
                raise ValueError("num_ranks must be specified either directly or via prompt_learner_cfg")

        self.embed_dims = embed_dims
        self.num_ranks = num_ranks

        # Build FER image encoder
        self.image_encoder = _build_fer_encoder(
            image_encoder_name, num_classes=embed_dims, pretrained_path=fer_weights_path
        )
        # Ensure input_resolution is set (required by some data transforms)
        if not hasattr(self.image_encoder, "input_resolution"):
            self.image_encoder.input_resolution = 224

        # Linear classification head (same as Baseline)
        self.last_project = nn.Linear(embed_dims, num_ranks, bias=False)

        # Null attributes for Runner compatibility
        self.text_encoder = None
        self.prompt_learner = None
        self.logit_scale = None

        # Loud warning if no FER weights -- experiments would be mislabeled
        self._has_fer_weights = fer_weights_path is not None
        if not self._has_fer_weights:
            logger.warning(
                "=" * 60 + "\n"
                "  WARNING: fer_weights_path is None!\n"
                "  The encoder is using ImageNet-only pretrained weights,\n"
                "  NOT FER-pretrained features. Do NOT label this experiment\n"
                "  as a 'FER baseline'. Set fer_weights_path to a valid\n"
                "  checkpoint or use HSEmotion auto-download.\n"
                + "=" * 60
            )

        logger.info(
            f"FERBaseline: encoder={image_encoder_name}, "
            f"embed_dims={embed_dims}, num_ranks={num_ranks}, "
            f"fer_weights={fer_weights_path or 'NONE (ImageNet only)'}"
        )

    def forward(self, images: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            images: [B, 3, H, W] input images.

        Returns:
            logits:          [B, num_ranks] classification logits.
            image_features:  [B, embed_dims] normalized image features.
            text_features:   None (no text encoder).
        """
        image_features = self.image_encoder(images)  # [B, D]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [B, D]

        text_features = self.last_project.weight  # [num_ranks, D]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [num_ranks, D]

        logits = 100.0 * image_features @ text_features.t()  # [B, num_ranks]
        return logits, image_features, None

    def forward_text_only(self) -> torch.Tensor:
        """Return learned class prototypes (for visualization/analysis)."""
        return self.last_project.weight  # [num_ranks, D]

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image features without classification."""
        return self.image_encoder(x)  # [B, D]

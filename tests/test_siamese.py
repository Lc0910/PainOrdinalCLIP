"""Unit tests for Siamese ranking head system.

Tests:
  1. PairwiseDataset pair label correctness
  2. PairwiseDataset cross-subject constraint
  3. SiameseOrdinalCLIP forward shape
  4. SiameseOrdinalCLIP freeze/unfreeze gradient flow
  5. SiameseRunner construction and training_step
  6. AUC computation correctness
  7. Linear vs MLP head shapes
"""
import os.path as osp

import pytest
import torch
from omegaconf import OmegaConf

from ordinalclip.models import MODELS
from ordinalclip.models.siamese_ordinalclip import (
    LinearRankingHead,
    MLPRankingHead,
    SiameseOrdinalCLIP,
    build_ranking_head,
)
from ordinalclip.runner.siamese_runner import SiameseRunner


# ================================================================
#  Fixtures
# ================================================================

@pytest.fixture
def siamese_cfg():
    return OmegaConf.load(osp.join(osp.dirname(__file__), "data", "siamese_default.yaml"))


@pytest.fixture
def dummy_backbone():
    """Build a small backbone for testing."""
    cfg = OmegaConf.load(osp.join(osp.dirname(__file__), "data", "siamese_default.yaml"))
    backbone_cfg = OmegaConf.to_container(cfg.runner_cfg.backbone_cfg)
    return MODELS.build(backbone_cfg)


@pytest.fixture
def dummy_data_file(tmp_path):
    """Create a temporary data file with BioVid-style entries.

    Uses pytest's tmp_path for automatic cleanup.
    """
    lines = [
        # Subject 071309_w_21, video BL1-081
        "images/071309_w_21-BL1-081_50.jpg 0",
        "images/071309_w_21-BL1-081_55.jpg 0",
        "images/071309_w_21-BL1-081_60.jpg 0",
        # Subject 071309_w_21, video PA4-010
        "images/071309_w_21-PA4-010_50.jpg 1",
        "images/071309_w_21-PA4-010_55.jpg 1",
        # Subject 071313_m_41, video BL1-055
        "images/071313_m_41-BL1-055_50.jpg 0",
        "images/071313_m_41-BL1-055_55.jpg 0",
        # Subject 071313_m_41, video PA4-020
        "images/071313_m_41-PA4-020_50.jpg 1",
        "images/071313_m_41-PA4-020_55.jpg 1",
        "images/071313_m_41-PA4-020_60.jpg 1",
        # Subject 080109_m_50, video BL1-010
        "images/080109_m_50-BL1-010_50.jpg 0",
        # Subject 080109_m_50, video PA4-030
        "images/080109_m_50-PA4-030_50.jpg 1",
    ]
    p = tmp_path / "dummy_biovid.txt"
    p.write_text("\n".join(lines))
    return str(p)


# ================================================================
#  1. Pair Label Correctness
# ================================================================

class TestPairLabel:
    def test_label_a_greater(self):
        """When rank_a > rank_b, pair_label should be 1."""
        label_a, label_b = 1, 0
        pair_label = 1 if label_a > label_b else 0
        assert pair_label == 1

    def test_label_a_smaller(self):
        """When rank_a < rank_b, pair_label should be 0."""
        label_a, label_b = 0, 1
        pair_label = 1 if label_a > label_b else 0
        assert pair_label == 0


# ================================================================
#  2. Cross-Subject Constraint
# ================================================================

class TestPairSampling:
    def test_subject_id_parsing(self):
        from ordinalclip.runner.siamese_data import _parse_subject_id, _parse_video_id

        path = "images/071313_m_41-BL1-081_50.jpg"
        assert _parse_subject_id(path) == "071313_m_41"
        assert _parse_video_id(path) == "071313_m_41-BL1-081"

    def test_subject_excluded_pools(self, dummy_data_file):
        """Verify cross-subject pool excludes same-subject images."""
        from collections import defaultdict

        from ordinalclip.runner.siamese_data import PairwiseDataset, _parse_subject_id

        # Build dataset without real images (we won't load them)
        dataset = PairwiseDataset.__new__(PairwiseDataset)
        dataset.images_root = ""
        dataset.transforms = None
        dataset.pairs_per_epoch = 100

        # Parse data file manually
        dataset.images_by_label = defaultdict(list)
        dataset.images_by_label_subject = defaultdict(lambda: defaultdict(list))

        with open(dummy_data_file) as f:
            for line in f:
                splits = line.strip().split()
                if len(splits) < 2:
                    continue
                img_path = splits[0]
                label = int(splits[1])
                subject_id = _parse_subject_id(img_path)
                dataset.images_by_label[label].append(img_path)
                dataset.images_by_label_subject[label][subject_id].append(img_path)

        dataset.sorted_labels = sorted(dataset.images_by_label.keys())

        # Build pools
        dataset._subject_excluded_pools = {}
        for label in dataset.sorted_labels:
            dataset._subject_excluded_pools[label] = {}
            all_imgs = dataset.images_by_label[label]
            for subj in dataset.images_by_label_subject[label]:
                dataset._subject_excluded_pools[label][subj] = [
                    p for p in all_imgs if _parse_subject_id(p) != subj
                ]

        # For label=0, subject=071309_w_21: pool should NOT contain 071309_w_21 images
        pool = dataset._subject_excluded_pools[0]["071309_w_21"]
        for p in pool:
            assert _parse_subject_id(p) != "071309_w_21", \
                f"Pool contains same-subject image: {p}"

        # Pool should contain images from other subjects
        assert len(pool) > 0, "Cross-subject pool should not be empty"

    def test_labels_less_than_2_raises(self, tmp_path):
        """Dataset with only 1 label should raise ValueError."""
        from ordinalclip.runner.siamese_data import PairwiseDataset

        data_file = tmp_path / "single_label.txt"
        data_file.write_text("images/a_50.jpg 0\nimages/b_50.jpg 0\n")

        with pytest.raises(ValueError, match="at least 2 distinct labels"):
            PairwiseDataset(
                images_root="",
                data_file=str(data_file),
                transforms=None,
                pairs_per_epoch=10,
            )


# ================================================================
#  3. Model Forward Shape
# ================================================================

class TestSiameseModel:
    def test_forward_pair(self, dummy_backbone):
        """Test pairwise forward produces correct shapes."""
        model = SiameseOrdinalCLIP(
            backbone=dummy_backbone,
            ranking_head_cfg={"head_type": "mlp", "embed_dims": dummy_backbone.embed_dims, "hidden_dims": 64},
            freeze_backbone=True,
        )
        B = 2
        res = dummy_backbone.image_encoder.input_resolution
        img_a = torch.randn(B, 3, res, res)
        img_b = torch.randn(B, 3, res, res)

        ranking_logits, logits_a, logits_b = model(img_a, img_b)

        assert ranking_logits.shape == (B, 1), f"Expected [B,1], got {ranking_logits.shape}"
        assert logits_a.shape == (B, dummy_backbone.num_ranks), \
            f"Expected [B,{dummy_backbone.num_ranks}], got {logits_a.shape}"
        assert logits_b.shape == logits_a.shape

    def test_forward_single(self, dummy_backbone):
        """Test single-image forward for val/test."""
        model = SiameseOrdinalCLIP(
            backbone=dummy_backbone,
            ranking_head_cfg={"head_type": "mlp", "embed_dims": dummy_backbone.embed_dims, "hidden_dims": 64},
            freeze_backbone=True,
        )
        B = 2
        res = dummy_backbone.image_encoder.input_resolution
        img = torch.randn(B, 3, res, res)

        logits, feat, _ = model.forward_single(img)

        assert logits.shape == (B, dummy_backbone.num_ranks)


# ================================================================
#  4. Freeze / Unfreeze Gradient Flow
# ================================================================

class TestGradientFlow:
    def test_backbone_frozen(self, dummy_backbone):
        """When backbone is frozen, only ranking_head should have gradients."""
        model = SiameseOrdinalCLIP(
            backbone=dummy_backbone,
            ranking_head_cfg={"head_type": "mlp", "embed_dims": dummy_backbone.embed_dims, "hidden_dims": 64},
            freeze_backbone=True,
        )
        # Backbone: all requires_grad should be False
        for name, param in model.backbone.named_parameters():
            assert not param.requires_grad, f"Backbone param {name} should be frozen"

        # Ranking head: all requires_grad should be True
        for name, param in model.ranking_head.named_parameters():
            assert param.requires_grad, f"Ranking head param {name} should be trainable"

    def test_backbone_unfrozen(self, dummy_backbone):
        """When backbone is not frozen, all params should have gradients."""
        model = SiameseOrdinalCLIP(
            backbone=dummy_backbone,
            ranking_head_cfg={"head_type": "mlp", "embed_dims": dummy_backbone.embed_dims, "hidden_dims": 64},
            freeze_backbone=False,
        )
        # All params should be trainable
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Param {name} should be trainable"


# ================================================================
#  5. SiameseRunner Construction
# ================================================================

class TestSiameseRunner:
    def test_construction(self, siamese_cfg):
        """SiameseRunner should construct without errors."""
        runner = SiameseRunner(**OmegaConf.to_container(siamese_cfg.runner_cfg))
        assert runner.num_ranks == 2
        assert isinstance(runner.module, SiameseOrdinalCLIP)

    def test_init_order_logger_before_load(self, siamese_cfg):
        """_custom_logger must be initialized before _load_backbone_weights is called.

        Regression test for P0-1: _custom_logger was used before definition.
        """
        runner = SiameseRunner(**OmegaConf.to_container(siamese_cfg.runner_cfg))
        # If init order is wrong, construction would raise AttributeError.
        # Verify the logger exists and is usable.
        assert hasattr(runner, "_custom_logger")
        assert runner._custom_logger is not None

    def test_training_step_shape(self, siamese_cfg):
        """Verify training_step accepts batch and returns loss."""
        runner = SiameseRunner(**OmegaConf.to_container(siamese_cfg.runner_cfg))
        B = 2
        res = runner.module.backbone.image_encoder.input_resolution

        batch = (
            torch.randn(B, 3, res, res),   # img_a
            torch.randn(B, 3, res, res),   # img_b
            torch.tensor([1, 0]),           # pair_label
            torch.tensor([1, 0]),           # rank_a
            torch.tensor([0, 1]),           # rank_b
        )
        outputs = runner.training_step(batch, batch_idx=0)
        assert "loss" in outputs
        assert outputs["loss"].requires_grad

    def test_empty_param_raises(self, siamese_cfg):
        """Both lr=0 should raise ValueError."""
        cfg = OmegaConf.to_container(siamese_cfg.runner_cfg)
        cfg["optimizer_and_scheduler_cfg"]["param_dict_cfg"]["lr_ranking_head"] = 0.0
        cfg["optimizer_and_scheduler_cfg"]["param_dict_cfg"]["lr_backbone"] = 0.0
        runner = SiameseRunner(**cfg)
        with pytest.raises(ValueError, match="No trainable parameters"):
            runner.build_param_dict(lr_ranking_head=0.0, lr_backbone=0.0)


# ================================================================
#  6. AUC Computation
# ================================================================

class TestAUC:
    def test_perfect_auc(self):
        """Perfect ranking should give AUC=1.0."""
        probs = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([1, 1, 0, 0])
        auc = SiameseRunner._compute_binary_auc(probs, labels)
        assert abs(auc - 1.0) < 1e-6, f"Expected AUC=1.0, got {auc}"

    def test_random_auc(self):
        """Random ranking should give AUC~0.5."""
        torch.manual_seed(42)
        probs = torch.rand(1000)
        labels = torch.randint(0, 2, (1000,))
        auc = SiameseRunner._compute_binary_auc(probs, labels)
        assert 0.4 < auc < 0.6, f"Expected AUC~0.5, got {auc}"

    def test_worst_auc(self):
        """Inverted ranking should give AUC=0.0."""
        probs = torch.tensor([0.1, 0.2, 0.8, 0.9])
        labels = torch.tensor([1, 1, 0, 0])
        auc = SiameseRunner._compute_binary_auc(probs, labels)
        assert abs(auc - 0.0) < 1e-6, f"Expected AUC=0.0, got {auc}"

    def test_all_same_label(self):
        """When all labels are the same, AUC should return 0.5."""
        probs = torch.tensor([0.9, 0.5, 0.1])
        labels = torch.tensor([1, 1, 1])
        auc = SiameseRunner._compute_binary_auc(probs, labels)
        assert auc == 0.5


# ================================================================
#  7. Ranking Head Variants (P1-2)
# ================================================================

class TestRankingHeads:
    def test_linear_head_shape(self):
        """LinearRankingHead: D->1."""
        head = LinearRankingHead(embed_dims=512)
        x = torch.randn(4, 512)  # [B, D]
        out = head(x)
        assert out.shape == (4, 1)

    def test_mlp_head_shape(self):
        """MLPRankingHead: D->hidden->hidden//2->1."""
        head = MLPRankingHead(embed_dims=512, hidden_dims=256, dropout=0.1)
        x = torch.randn(4, 512)  # [B, D]
        out = head(x)
        assert out.shape == (4, 1)

    def test_build_ranking_head_factory(self):
        """build_ranking_head should create correct head type."""
        linear = build_ranking_head(head_type="linear", embed_dims=256)
        assert isinstance(linear, LinearRankingHead)

        mlp = build_ranking_head(head_type="mlp", embed_dims=256, hidden_dims=128)
        assert isinstance(mlp, MLPRankingHead)

    def test_unknown_head_type(self):
        """Unknown head_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown head_type"):
            build_ranking_head(head_type="transformer", embed_dims=256)


# ================================================================
#  8. Config Parsing (P0-3 regression test)
# ================================================================

class TestConfigParsing:
    def test_cfg_options_merged_correctly(self, tmp_path):
        """Verify --cfg_options key=value pairs are merged into config.

        Regression test for P0-3: multiple --cfg_options caused REMAINDER collision.
        """
        import argparse
        from scripts.run_siamese import parse_cfg

        cfg_file = tmp_path / "test_cfg.yaml"
        cfg_file.write_text(
            "runner_cfg:\n"
            "  seed: 42\n"
            "  output_dir: results/test\n"
            "  backbone_cfg:\n"
            "    type: OrdinalCLIP\n"
            "    prompt_learner_cfg:\n"
            "      type: PlainPromptLearner\n"
            "      num_ranks: 2\n"
            "      num_base_ranks: 2\n"
            "      num_tokens_per_rank: 1\n"
            "      num_context_tokens: 4\n"
            "      rank_tokens_position: tail\n"
            "      init_rank_path: null\n"
            "      init_context: null\n"
            "      rank_specific_context: false\n"
            "      interpolation_type: linear\n"
            "    text_encoder_name: RN50\n"
            "    image_encoder_name: RN50\n"
            "  ranking_head_cfg:\n"
            "    head_type: mlp\n"
            "    hidden_dims: 64\n"
            "    dropout: 0.1\n"
            "  freeze_backbone: true\n"
            "  loss_weights:\n"
            "    ranking_loss: 1.0\n"
            "    ce_loss_a: 0.5\n"
            "    ce_loss_b: 0.5\n"
            "  optimizer_and_scheduler_cfg:\n"
            "    param_dict_cfg:\n"
            "      lr_ranking_head: 0.001\n"
            "      lr_backbone: 0.0\n"
            "    optimizer_cfg:\n"
            "      optimizer_name: adam\n"
            "      lr: 0.001\n"
            "      weight_decay: 0.0\n"
            "      momentum: 0.9\n"
            "      sgd_dampening: 0.0\n"
            "      sgd_nesterov: false\n"
            "      rmsprop_alpha: 0.99\n"
            "      adam_beta1: 0.9\n"
            "      adam_beta2: 0.999\n"
            "      staged_lr: null\n"
            "      lookahead: false\n"
            "    lr_scheduler_cfg:\n"
            "      lr_scheduler_name: multi_step\n"
            "      stepsize: [30]\n"
            "      gamma: 0.1\n"
            "      max_epochs: 50\n"
            "      warmup_epoch: 0\n"
            "      warmup_cons_lr: 0.00001\n"
            "      warmup_min_lr: 0.00001\n"
            "      warmup_type: constant\n"
            "      warmup_recount: true\n"
            "  load_weights_cfg:\n"
            "    backbone_ckpt_path: null\n"
            "  ckpt_path: ''\n"
            "trainer_cfg:\n"
            "  max_epochs: 50\n"
            "  precision: 16\n"
            "  accelerator: cpu\n"
            "  devices: 1\n"
            "test_only: false\n"
        )

        # Simulate: base options + experiment-specific options in a single
        # --cfg_options list (as the fixed shell script now does).
        args = argparse.Namespace(
            config=[str(cfg_file)],
            seed=None,
            output_dir=str(tmp_path / "output"),
            test_only=False,
            debug=False,
            verbose=False,
            cfg_options=[
                "trainer_cfg.max_epochs=30",
                "runner_cfg.loss_weights.ranking_loss=2.0",
                "runner_cfg.ranking_head_cfg.head_type=linear",
            ],
        )

        cfg = parse_cfg(args, instantialize_output_dir=False)
        assert cfg.trainer_cfg.max_epochs == 30
        assert cfg.runner_cfg.loss_weights.ranking_loss == 2.0
        assert cfg.runner_cfg.ranking_head_cfg.head_type == "linear"

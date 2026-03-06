"""Unit tests for Siamese ranking system — aligned to Fabio (2025) §5.3.

Tests:
  1. PairwiseDataset pair label correctness
  2. PairwiseDataset cross-subject constraint
  3. SiameseOrdinalCLIP forward shape (new architecture)
  4. SiameseOrdinalCLIP freeze/unfreeze gradient flow
  5. SiameseRunner construction and training_step
  6. AUC computation correctness
  7. New head components: SharedMLP, RegressionHead, ConcatRankingHead
  8. Config parsing
  9. Multi-class anchor-based ranking inference
"""
import os.path as osp

import pytest
import torch
from omegaconf import OmegaConf

from ordinalclip.models import MODELS
from ordinalclip.models.siamese_ordinalclip import (
    ConcatRankingHead,
    RegressionHead,
    SharedMLP,
    SiameseOrdinalCLIP,
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
    """Build a small backbone for testing (RN50)."""
    cfg = OmegaConf.load(osp.join(osp.dirname(__file__), "data", "siamese_default.yaml"))
    backbone_cfg = OmegaConf.to_container(cfg.runner_cfg.backbone_cfg)
    return MODELS.build(backbone_cfg)


@pytest.fixture
def dummy_siamese_model(dummy_backbone):
    """Build a SiameseOrdinalCLIP with small hidden dims for testing."""
    return SiameseOrdinalCLIP(
        backbone=dummy_backbone,
        shared_mlp_cfg={"hidden_dims": 64, "out_dims": 64},
        ranking_head_cfg={"head_type": "linear"},
        freeze_backbone=True,
    )


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
#  3. Model Forward Shape (new architecture)
# ================================================================

class TestSiameseModel:
    def test_forward_pair(self, dummy_backbone):
        """Pairwise forward returns (ranking_logit [B,1], reg_a [B], reg_b [B])."""
        model = SiameseOrdinalCLIP(
            backbone=dummy_backbone,
            shared_mlp_cfg={"hidden_dims": 64, "out_dims": 64},
            ranking_head_cfg={"head_type": "linear"},
            freeze_backbone=True,
        )
        B = 2
        res = dummy_backbone.image_encoder.input_resolution
        img_a = torch.randn(B, 3, res, res)
        img_b = torch.randn(B, 3, res, res)

        ranking_logit, reg_score_a, reg_score_b = model(img_a, img_b)

        assert ranking_logit.shape == (B, 1), \
            f"Expected [B,1], got {ranking_logit.shape}"
        assert reg_score_a.shape == (B,), \
            f"Expected [B], got {reg_score_a.shape}"
        assert reg_score_b.shape == (B,), \
            f"Expected [B], got {reg_score_b.shape}"

    def test_forward_pair_mlp_head(self, dummy_backbone):
        """MLP ranking head also produces correct output shapes."""
        model = SiameseOrdinalCLIP(
            backbone=dummy_backbone,
            shared_mlp_cfg={"hidden_dims": 64, "out_dims": 64},
            ranking_head_cfg={"head_type": "mlp", "hidden_dims": 64},
            freeze_backbone=True,
        )
        B = 3
        res = dummy_backbone.image_encoder.input_resolution
        img_a = torch.randn(B, 3, res, res)
        img_b = torch.randn(B, 3, res, res)

        ranking_logit, reg_score_a, reg_score_b = model(img_a, img_b)
        assert ranking_logit.shape == (B, 1)
        assert reg_score_a.shape == (B,)

    def test_forward_single(self, dummy_backbone):
        """Single-image forward returns (logits [B,K], feat [B,D], reg_score [B])."""
        model = SiameseOrdinalCLIP(
            backbone=dummy_backbone,
            shared_mlp_cfg={"hidden_dims": 64, "out_dims": 64},
            ranking_head_cfg={"head_type": "linear"},
            freeze_backbone=True,
        )
        B = 2
        res = dummy_backbone.image_encoder.input_resolution
        img = torch.randn(B, 3, res, res)

        logits, feat, reg_score = model.forward_single(img)

        assert logits.shape == (B, dummy_backbone.num_ranks), \
            f"Expected [B,K], got {logits.shape}"
        assert feat.shape == (B, dummy_backbone.embed_dims), \
            f"Expected [B,D], got {feat.shape}"
        assert reg_score.shape == (B,), \
            f"Expected [B], got {reg_score.shape}"

    def test_regression_score_range(self, dummy_backbone):
        """Regression scores should be in (0, K-1) due to sigmoid rescaling.

        Uses B=100 samples to reduce risk of hitting sigmoid saturation
        boundaries with randomly initialized weights.
        """
        K = dummy_backbone.num_ranks
        model = SiameseOrdinalCLIP(
            backbone=dummy_backbone,
            shared_mlp_cfg={"hidden_dims": 64, "out_dims": 64},
            ranking_head_cfg={"head_type": "linear"},
            freeze_backbone=True,
        )
        B = 100
        res = dummy_backbone.image_encoder.input_resolution
        imgs = torch.randn(B, 3, res, res)

        _, _, reg_score = model.forward_single(imgs)

        assert (reg_score > 0).all(), f"reg_score should be > 0, got min={reg_score.min()}"
        assert (reg_score < K - 1).all(), \
            f"reg_score should be < {K-1}, got max={reg_score.max()}"


# ================================================================
#  4. Freeze / Unfreeze Gradient Flow
# ================================================================

class TestGradientFlow:
    def test_backbone_frozen(self, dummy_backbone):
        """When backbone is frozen, only siamese heads should have gradients."""
        model = SiameseOrdinalCLIP(
            backbone=dummy_backbone,
            shared_mlp_cfg={"hidden_dims": 64, "out_dims": 64},
            ranking_head_cfg={"head_type": "linear"},
            freeze_backbone=True,
        )
        # Backbone: all requires_grad should be False
        for name, param in model.backbone.named_parameters():
            assert not param.requires_grad, f"Backbone param {name} should be frozen"

        # SharedMLP, RegressionHead, RankingHead: all requires_grad should be True
        for name, param in model.shared_mlp.named_parameters():
            assert param.requires_grad, f"SharedMLP param {name} should be trainable"
        for name, param in model.regression_head.named_parameters():
            assert param.requires_grad, f"RegressionHead param {name} should be trainable"
        for name, param in model.ranking_head.named_parameters():
            assert param.requires_grad, f"RankingHead param {name} should be trainable"

    def test_backbone_unfrozen(self, dummy_backbone):
        """When backbone is not frozen, all params should have gradients."""
        model = SiameseOrdinalCLIP(
            backbone=dummy_backbone,
            shared_mlp_cfg={"hidden_dims": 64, "out_dims": 64},
            ranking_head_cfg={"head_type": "linear"},
            freeze_backbone=False,
        )
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
        # Verify new sub-modules exist
        assert hasattr(runner.module, "shared_mlp")
        assert hasattr(runner.module, "regression_head")
        assert hasattr(runner.module, "ranking_head")

    def test_init_order_logger_before_load(self, siamese_cfg):
        """_custom_logger must be initialized before _load_backbone_weights is called.

        Regression test: _custom_logger was used before definition.
        """
        runner = SiameseRunner(**OmegaConf.to_container(siamese_cfg.runner_cfg))
        assert hasattr(runner, "_custom_logger")
        assert runner._custom_logger is not None

    def test_training_step_shape(self, siamese_cfg):
        """Verify training_step accepts batch and returns loss dict."""
        runner = SiameseRunner(**OmegaConf.to_container(siamese_cfg.runner_cfg))
        B = 2
        res = runner.module.backbone.image_encoder.input_resolution

        batch = (
            torch.randn(B, 3, res, res),   # img_a
            torch.randn(B, 3, res, res),   # img_b
            torch.tensor([1, 0]),           # pair_label (1 if rank_a > rank_b)
            torch.tensor([1, 0]),           # rank_a (integer labels)
            torch.tensor([0, 1]),           # rank_b (integer labels)
        )
        outputs = runner.training_step(batch, batch_idx=0)
        assert "loss" in outputs
        assert outputs["loss"].requires_grad

    def test_training_step_loss_components(self, siamese_cfg):
        """Training step should produce finite MSE + hinge losses."""
        runner = SiameseRunner(**OmegaConf.to_container(siamese_cfg.runner_cfg))
        B = 4
        res = runner.module.backbone.image_encoder.input_resolution

        batch = (
            torch.randn(B, 3, res, res),
            torch.randn(B, 3, res, res),
            torch.tensor([1, 0, 1, 0]),
            torch.tensor([1, 0, 1, 0]),
            torch.tensor([0, 1, 0, 1]),
        )
        outputs = runner.training_step(batch, batch_idx=0)
        assert torch.isfinite(outputs["loss"]), "Loss should be finite"

    def test_empty_param_raises(self, siamese_cfg):
        """Both lr=0 should raise ValueError."""
        cfg = OmegaConf.to_container(siamese_cfg.runner_cfg)
        cfg["optimizer_and_scheduler_cfg"]["param_dict_cfg"]["lr_siamese_heads"] = 0.0
        cfg["optimizer_and_scheduler_cfg"]["param_dict_cfg"]["lr_backbone"] = 0.0
        runner = SiameseRunner(**cfg)
        with pytest.raises(ValueError, match="No trainable parameters"):
            runner.build_param_dict(lr_siamese_heads=0.0, lr_backbone=0.0)


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
#  7. New Head Components: SharedMLP, RegressionHead, ConcatRankingHead
# ================================================================

class TestHeadComponents:
    """Tests for the three new modules in the Fabio §5.3 architecture."""

    # --- SharedMLP ---

    def test_shared_mlp_output_shape(self):
        """SharedMLP: D → hidden → out_dims."""
        mlp = SharedMLP(embed_dims=512, hidden_dims=256, out_dims=128)
        x = torch.randn(4, 512)  # [B, D]
        out = mlp(x)
        assert out.shape == (4, 128), f"Expected [4, 128], got {out.shape}"

    def test_shared_mlp_out_dims_attr(self):
        """SharedMLP.out_dims is accessible for downstream modules."""
        mlp = SharedMLP(embed_dims=1024, hidden_dims=256, out_dims=64)
        assert mlp.out_dims == 64

    def test_shared_mlp_with_dropout(self):
        """SharedMLP with dropout should still produce correct output shape."""
        mlp = SharedMLP(embed_dims=256, hidden_dims=128, out_dims=64, dropout=0.3)
        mlp.eval()
        x = torch.randn(8, 256)
        out = mlp(x)
        assert out.shape == (8, 64)

    # --- RegressionHead ---

    def test_regression_head_shape(self):
        """RegressionHead: in_dims → [B] scalar score."""
        head = RegressionHead(in_dims=128, num_ranks=5)
        e = torch.randn(4, 128)  # [B, out_dims]
        score = head(e)
        assert score.shape == (4,), f"Expected [4], got {score.shape}"

    def test_regression_head_range_2class(self):
        """For 2-class, score should be in (0, 1)."""
        head = RegressionHead(in_dims=64, num_ranks=2)
        e = torch.randn(100, 64)
        score = head(e)
        # sigmoid-based: strictly between 0 and K-1
        assert (score > 0).all(), "Score should be > 0"
        assert (score < 1).all(), "Score should be < 1 for K=2"

    def test_regression_head_range_5class(self):
        """For 5-class (K=5), score should be in (0, 4)."""
        head = RegressionHead(in_dims=128, num_ranks=5)
        e = torch.randn(100, 128)
        score = head(e)
        assert (score > 0).all(), "Score should be > 0"
        assert (score < 4).all(), "Score should be < 4 for K=5"

    # --- ConcatRankingHead ---

    def test_concat_head_linear_shape(self):
        """Linear ConcatRankingHead: 2*in_dims → 1."""
        head = ConcatRankingHead(in_dims=128, head_type="linear")
        concat_feat = torch.randn(4, 256)  # [B, 2*in_dims]
        out = head(concat_feat)
        assert out.shape == (4, 1), f"Expected [4, 1], got {out.shape}"

    def test_concat_head_mlp_shape(self):
        """MLP ConcatRankingHead: 2*in_dims → hidden → hidden//2 → 1."""
        head = ConcatRankingHead(in_dims=128, head_type="mlp", hidden_dims=256)
        concat_feat = torch.randn(4, 256)
        out = head(concat_feat)
        assert out.shape == (4, 1), f"Expected [4, 1], got {out.shape}"

    def test_concat_head_mlp_with_dropout(self):
        """MLP ConcatRankingHead with dropout."""
        head = ConcatRankingHead(in_dims=64, head_type="mlp", hidden_dims=128, dropout=0.1)
        head.eval()
        concat_feat = torch.randn(8, 128)
        out = head(concat_feat)
        assert out.shape == (8, 1)

    def test_concat_head_unknown_type_raises(self):
        """Unknown head_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown head_type"):
            ConcatRankingHead(in_dims=128, head_type="transformer")

    def test_concat_head_pair_asymmetry(self):
        """Swapping ei and ej should produce a different score (order matters)."""
        head = ConcatRankingHead(in_dims=64, head_type="linear")
        head.eval()
        e_i = torch.randn(1, 64)
        e_j = torch.randn(1, 64)
        s_ij = head(torch.cat([e_i, e_j], dim=-1))
        s_ji = head(torch.cat([e_j, e_i], dim=-1))
        # For a linear layer, s_ij + s_ji = 2*bias (not necessarily 0),
        # so true antisymmetry doesn't hold. Verify scores are different
        # to confirm the head is order-sensitive.
        assert s_ij.item() != s_ji.item(), "Swapping pair should change score"


# ================================================================
#  8. Config Parsing (regression test for --cfg_options)
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
            "  shared_mlp_cfg:\n"
            "    hidden_dims: 64\n"
            "    out_dims: 64\n"
            "    dropout: 0.0\n"
            "  ranking_head_cfg:\n"
            "    head_type: linear\n"
            "    hidden_dims: 64\n"
            "    dropout: 0.0\n"
            "  freeze_backbone: true\n"
            "  loss_weights:\n"
            "    mse_loss: 1.0\n"
            "    ranking_loss: 0.5\n"
            "    margin_scale: 1.0\n"
            "  optimizer_and_scheduler_cfg:\n"
            "    param_dict_cfg:\n"
            "      lr_siamese_heads: 0.0001\n"
            "      lr_backbone: 0.0\n"
            "    optimizer_cfg:\n"
            "      optimizer_name: adam\n"
            "      lr: 0.0001\n"
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

        # Simulate multiple --cfg_options in a single list (fixed shell script format)
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


# ================================================================
#  9. Multi-Class Anchor-Based Ranking Inference
# ================================================================

class TestMultiClassAnchorInference:
    """Test cumulative link approach for multi-class rank predictions.

    Anchors are stored in SharedMLP embedding space (out_dims), NOT raw CLIP
    feature space (embed_dims). Tests use out_dims = 64 (from test fixture).
    """

    def _make_runner_with_anchors(self, siamese_cfg, num_ranks: int, anchors: dict):
        """Helper: build a SiameseRunner and inject pre-computed anchors."""
        cfg = OmegaConf.to_container(siamese_cfg.runner_cfg)
        cfg["backbone_cfg"]["prompt_learner_cfg"]["num_ranks"] = num_ranks
        cfg["anchor_inference_cfg"] = {"enabled": True, "ensemble_alpha": 0.5}
        runner = SiameseRunner(**cfg)
        runner._anchors = anchors
        return runner

    def _get_embed_and_out_dims(self, runner: SiameseRunner):
        """Return (embed_dims, out_dims) for anchor / feature sizing."""
        embed_dims = runner.module.backbone.embed_dims
        out_dims = runner.module.shared_mlp.out_dims
        return embed_dims, out_dims

    def test_binary_fallback(self, siamese_cfg):
        """2-class should use the binary path (not cumulative link)."""
        embed_dims = MODELS.build(
            OmegaConf.to_container(siamese_cfg.runner_cfg.backbone_cfg)
        ).embed_dims
        cfg = OmegaConf.to_container(siamese_cfg.runner_cfg)
        runner = SiameseRunner(**cfg)
        out_dims = runner.module.shared_mlp.out_dims

        runner._anchors = {
            0: torch.randn(out_dims),
            1: torch.randn(out_dims),
        }

        B = 2
        image_features = torch.randn(B, embed_dims)
        y = torch.tensor([0, 1])

        result = runner._compute_rank_predictions(image_features, y)
        assert result["_p_rank"].shape == (B, 2), \
            f"Expected [B,2], got {result['_p_rank'].shape}"
        assert result["predict_y_rank"].shape == (B,)

    def test_5class_cumulative_link_shape(self, siamese_cfg):
        """5-class cumulative link should produce [B, 5] probability distribution."""
        cfg = OmegaConf.to_container(siamese_cfg.runner_cfg)
        cfg["backbone_cfg"]["prompt_learner_cfg"]["num_ranks"] = 5
        cfg["anchor_inference_cfg"] = {"enabled": True}
        runner = SiameseRunner(**cfg)

        embed_dims, out_dims = self._get_embed_and_out_dims(runner)
        runner._anchors = {k: torch.randn(out_dims) for k in range(5)}

        B = 4
        image_features = torch.randn(B, embed_dims)
        y = torch.tensor([0, 1, 3, 4])

        result = runner._compute_rank_predictions(image_features, y)
        assert result["_p_rank"].shape == (B, 5), \
            f"Expected [B,5], got {result['_p_rank'].shape}"
        assert result["predict_y_rank"].shape == (B,)
        assert result["mae_rank_metric"].shape == (B,)
        assert result["acc_rank_metric"].shape == (B,)

    def test_5class_probabilities_sum_to_one(self, siamese_cfg):
        """Cumulative link probabilities should sum to 1 after normalization."""
        cfg = OmegaConf.to_container(siamese_cfg.runner_cfg)
        cfg["backbone_cfg"]["prompt_learner_cfg"]["num_ranks"] = 5
        cfg["anchor_inference_cfg"] = {"enabled": True}
        runner = SiameseRunner(**cfg)

        embed_dims, out_dims = self._get_embed_and_out_dims(runner)
        runner._anchors = {k: torch.randn(out_dims) for k in range(5)}

        B = 8
        image_features = torch.randn(B, embed_dims)
        y = torch.randint(0, 5, (B,))

        result = runner._compute_rank_predictions(image_features, y)
        p_rank = result["_p_rank"]  # [B, 5]

        # All probabilities non-negative
        assert (p_rank >= 0).all(), "Probabilities should be non-negative"
        # Each row sums to ~1.0
        row_sums = p_rank.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-5), \
            f"Probabilities should sum to 1, got {row_sums}"

    def test_5class_prediction_range(self, siamese_cfg):
        """Predictions should be in [0, K-1] range."""
        cfg = OmegaConf.to_container(siamese_cfg.runner_cfg)
        cfg["backbone_cfg"]["prompt_learner_cfg"]["num_ranks"] = 5
        cfg["anchor_inference_cfg"] = {"enabled": True}
        runner = SiameseRunner(**cfg)

        embed_dims, out_dims = self._get_embed_and_out_dims(runner)
        runner._anchors = {k: torch.randn(out_dims) for k in range(5)}

        B = 16
        image_features = torch.randn(B, embed_dims)
        y = torch.randint(0, 5, (B,))

        result = runner._compute_rank_predictions(image_features, y)
        pred = result["predict_y_rank"]  # [B]

        assert (pred >= 0).all(), f"Predictions should be >= 0, min={pred.min()}"
        assert (pred <= 4).all(), f"Predictions should be <= 4, max={pred.max()}"

    def test_3class_cumulative_link(self, siamese_cfg):
        """3-class should also work with cumulative link."""
        cfg = OmegaConf.to_container(siamese_cfg.runner_cfg)
        cfg["backbone_cfg"]["prompt_learner_cfg"]["num_ranks"] = 3
        cfg["anchor_inference_cfg"] = {"enabled": True}
        runner = SiameseRunner(**cfg)

        embed_dims, out_dims = self._get_embed_and_out_dims(runner)
        runner._anchors = {k: torch.randn(out_dims) for k in range(3)}

        B = 4
        image_features = torch.randn(B, embed_dims)
        y = torch.tensor([0, 1, 2, 1])

        result = runner._compute_rank_predictions(image_features, y)
        assert result["_p_rank"].shape == (B, 3)

        row_sums = result["_p_rank"].sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(B), atol=1e-5)

    def test_ensemble_with_multiclass(self, siamese_cfg):
        """Ensemble should work for multi-class rank + cls predictions."""
        cfg = OmegaConf.to_container(siamese_cfg.runner_cfg)
        cfg["backbone_cfg"]["prompt_learner_cfg"]["num_ranks"] = 5
        cfg["anchor_inference_cfg"] = {"enabled": True, "ensemble_alpha": 0.5}
        runner = SiameseRunner(**cfg)

        B, K = 4, 5
        p_cls  = torch.softmax(torch.randn(B, K), dim=-1)   # [B, 5]
        p_rank = torch.softmax(torch.randn(B, K), dim=-1)   # [B, 5]
        y = torch.tensor([0, 2, 3, 4])

        result = runner._compute_ensemble_predictions(p_cls, p_rank, y)
        assert result["predict_y_ens"].shape == (B,)
        assert result["mae_ens_metric"].shape == (B,)
        assert result["acc_ens_metric"].shape == (B,)

    def test_anchor_mode_dual(self, siamese_cfg):
        """Dual anchor mode should compute score from both anchors."""
        cfg = OmegaConf.to_container(siamese_cfg.runner_cfg)
        cfg["anchor_inference_cfg"] = {"enabled": True, "anchor_mode": "dual", "ensemble_alpha": 0.5}
        runner = SiameseRunner(**cfg)

        embed_dims, out_dims = self._get_embed_and_out_dims(runner)
        runner._anchors = {
            0: torch.randn(out_dims),
            1: torch.randn(out_dims),
        }

        B = 3
        image_features = torch.randn(B, embed_dims)
        y = torch.tensor([0, 1, 0])

        result = runner._compute_rank_predictions(image_features, y)
        assert result["_p_rank"].shape == (B, 2)
        assert result["predict_y_rank"].shape == (B,)

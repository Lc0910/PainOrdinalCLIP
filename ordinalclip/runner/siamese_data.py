"""Pairwise dataset and data module for Siamese ranking training.

Key design (P0-2 compliance):
  - Pairs are ALWAYS cross-class: image_A and image_B have different labels.
  - Pairs are cross-subject by default: image_A and image_B come from different
    subjects via pre-built subject-excluded pools.
  - Graceful fallback for degenerate edge cases (e.g., a label has only one
    subject): first falls back to cross-video (different clip), then to any
    image from the target label. A warning is logged when this occurs.
  - This prevents data leakage from near-identical frames within the same clip
    and ensures subject-independent generalization in normal conditions.

BioVid filename convention:
    images/071313_m_41-BL1-081_50.jpg
    ^^^^^^^^^^^^^^^^  subject_id
                     ^^^^^^^^^^^  clip_id (stimulus info)
                                 ^^  frame_index
    video_id = "071313_m_41-BL1-081"
    subject_id = "071313_m_41"   (before first '-')
"""
from __future__ import annotations

import os
import os.path as osp
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ordinalclip.utils.logging import get_logger

from .utils import get_transforms

logger = get_logger(__name__)


def _parse_video_id(img_path: str) -> str:
    """Extract video_id by stripping trailing frame index.

    'images/071313_m_41-BL1-081_50.jpg' -> '071313_m_41-BL1-081'
    """
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return stem.rsplit("_", 1)[0]


def _parse_subject_id(img_path: str) -> str:
    """Extract subject_id (portion before first '-').

    'images/071313_m_41-BL1-081_50.jpg' -> '071313_m_41'
    """
    video_id = _parse_video_id(img_path)
    return video_id.split("-")[0]


class PairwiseDataset(Dataset):
    """Online pairwise sampling dataset for Siamese ranking training.

    Each __getitem__ returns:
        img_a:      [3, H, W]
        img_b:      [3, H, W]
        pair_label: 1 if rank_a > rank_b, 0 if rank_a < rank_b
        rank_a:     int (original label of A)
        rank_b:     int (original label of B)

    Sampling constraints (P0-2):
        - A and B MUST have different labels (cross-class).
        - A and B SHOULD come from different subjects (cross-subject).
          Fallback: cross-video if cross-subject pool is empty for the
          sampled (label, subject) pair; last-resort: any image from
          the target label (degenerate single-subject-per-label case).
    """

    def __init__(
        self,
        images_root: str,
        data_file: str,
        transforms,
        pairs_per_epoch: int = 10000,
    ):
        self.images_root = images_root
        self.transforms = transforms
        self.pairs_per_epoch = pairs_per_epoch

        # Parse data file and group by (label, subject_id)
        self.images_by_label: Dict[int, List[str]] = defaultdict(list)
        self.images_by_label_subject: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        with open(data_file) as f:
            for line in f:
                splits = line.strip().split()
                if len(splits) < 2:
                    continue
                img_path = splits[0]
                label = int(splits[1])
                subject_id = _parse_subject_id(img_path)
                self.images_by_label[label].append(img_path)
                self.images_by_label_subject[label][subject_id].append(img_path)

        self.sorted_labels = sorted(self.images_by_label.keys())
        if len(self.sorted_labels) < 2:
            raise ValueError(
                f"PairwiseDataset requires at least 2 distinct labels, "
                f"got {self.sorted_labels} in {data_file}"
            )
        self.name = osp.splitext(osp.basename(data_file))[0].lower()

        # Build subject-excluded pools: for each (label, subject), pool = all images
        # of that label EXCEPT from that subject.  Used to guarantee cross-subject.
        self._subject_excluded_pools: Dict[int, Dict[str, List[str]]] = {}
        for label in self.sorted_labels:
            self._subject_excluded_pools[label] = {}
            all_imgs = self.images_by_label[label]
            for subj in self.images_by_label_subject[label]:
                self._subject_excluded_pools[label][subj] = [
                    p for p in all_imgs if _parse_subject_id(p) != subj
                ]

        total = sum(len(v) for v in self.images_by_label.values())
        n_subjects = len(set(
            _parse_subject_id(p) for imgs in self.images_by_label.values() for p in imgs
        ))
        logger.info(
            f"PairwiseDataset [{self.name}]: {total} images, "
            f"{len(self.sorted_labels)} classes, {n_subjects} subjects, "
            f"{self.pairs_per_epoch} pairs/epoch"
        )

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, index: int):
        # 1. Sample two different labels
        label_a, label_b = random.sample(self.sorted_labels, 2)

        # 2. Sample image A
        img_path_a = random.choice(self.images_by_label[label_a])
        subject_a = _parse_subject_id(img_path_a)
        video_a = _parse_video_id(img_path_a)

        # 3. Sample image B: cross-subject from A (with graceful fallback)
        pool_b = self._subject_excluded_pools[label_b].get(subject_a)
        if not pool_b:
            # Fallback 1: cross-video only (subject pool empty for this combo)
            pool_b = [
                p for p in self.images_by_label[label_b]
                if _parse_video_id(p) != video_a
            ]
            if pool_b:
                logger.debug(
                    f"Cross-subject pool empty for label={label_b}, "
                    f"subject={subject_a}; fell back to cross-video."
                )
        if not pool_b:
            # Fallback 2: any image from label_b (degenerate case)
            pool_b = self.images_by_label[label_b]
            logger.warning(
                f"Cross-video pool also empty for label={label_b}, "
                f"video={video_a}; using unconstrained fallback."
            )

        img_path_b = random.choice(pool_b)

        # 4. Load images
        img_a = self._load_image(img_path_a)
        img_b = self._load_image(img_path_b)

        # 5. Pair label: 1 if A > B, 0 if A < B
        pair_label = 1 if label_a > label_b else 0

        return img_a, img_b, pair_label, label_a, label_b

    def _load_image(self, img_path: str) -> torch.Tensor:
        full_path = os.path.join(self.images_root, img_path)
        img = Image.open(full_path)
        if img.mode == "L":
            img = img.convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img


class SiameseDataModule(pl.LightningDataModule):
    """Data module for Siamese Stage 2.

    - Train: PairwiseDataset (online pair sampling, cross-subject)
    - Val/Test: original RegressionDataset (frame-level + video aggregation)
    """

    def __init__(
        self,
        train_images_root: str,
        train_data_file: str,
        val_images_root: str,
        val_data_file: str,
        test_images_root: str,
        test_data_file: str,
        transforms_cfg: dict,
        train_dataloder_cfg: dict,
        eval_dataloder_cfg: dict,
        pairs_per_epoch: int = 10000,
        # Ignored fields from parent data_cfg (for compatibility)
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            ignored_keys = [k for k in kwargs if k not in ("few_shot", "label_distributed_shift", "use_long_tail")]
            logger.info(
                f"SiameseDataModule: ignoring data_cfg keys: {list(kwargs.keys())}"
            )
            if ignored_keys:
                logger.warning(
                    f"SiameseDataModule: unexpected extra keys: {ignored_keys}"
                )
        train_transforms, eval_transforms = get_transforms(**transforms_cfg)

        self.train_set = PairwiseDataset(
            train_images_root,
            train_data_file,
            train_transforms,
            pairs_per_epoch=pairs_per_epoch,
        )

        from .data import RegressionDataset

        self.val_set = RegressionDataset(val_images_root, val_data_file, eval_transforms)
        self.test_set = RegressionDataset(test_images_root, test_data_file, eval_transforms)

        self.train_dataloder_cfg = train_dataloder_cfg
        self.eval_dataloder_cfg = eval_dataloder_cfg

        # Store references for anchor computation (single-image train loader)
        self._train_images_root = train_images_root
        self._train_data_file = train_data_file
        self._eval_transforms = eval_transforms

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.train_dataloder_cfg)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.eval_dataloder_cfg)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.eval_dataloder_cfg)

    def anchor_dataloader(self) -> DataLoader:
        """Single-image DataLoader from training data with eval transforms.

        Used for computing per-class feature centroids (anchors) for
        anchor-based ranking inference. Uses eval transforms (no augmentation)
        so extracted features are deterministic.
        """
        from .data import RegressionDataset

        anchor_set = RegressionDataset(
            self._train_images_root,
            self._train_data_file,
            self._eval_transforms,
        )
        return DataLoader(dataset=anchor_set, **self.eval_dataloder_cfg)

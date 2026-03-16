"""AU-Only Baseline for BioVid 5-Class Pain Classification.

Diagnostic experiment: determine if AU features from py-feat have
discriminative power independent of CLIP visual features.

Methods:
  - SVM (RBF + Linear) via sklearn
  - MLP (3-layer) via plain PyTorch

Usage:
    # 17-AU (all AUs)
    python scripts/diagnosis/au_only_baseline.py \
        --au_npz data/biovid/au_features_all17.npz --au_dim 17 \
        --output_dir results/au-only-baseline-all17

    # 8-AU (PSPI pain-relevant)
    python scripts/diagnosis/au_only_baseline.py \
        --au_npz data/biovid/au_features_pain8.npz --au_dim 8 \
        --output_dir results/au-only-baseline-pain8
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_au_dataset(
    npz_path: str,
    data_list_path: str,
    au_dim: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load AU features aligned with a data list file.

    Returns:
        features: [N, au_dim] float32
        labels:   [N] int64
        paths:    list of N image path strings
    """
    raw = np.load(npz_path, allow_pickle=False)
    npz_data: Dict[str, np.ndarray] = {k: raw[k] for k in raw.files}
    logger.info(f"NPZ loaded: {len(npz_data)} entries from {npz_path}")

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []
    paths_list: List[str] = []
    n_missing = 0
    n_total_lines = 0
    first_missing_key: Optional[str] = None

    with open(data_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_path = parts[0]
            label = int(parts[1])
            n_total_lines += 1

            au_vec = npz_data.get(img_path)
            if au_vec is None:
                n_missing += 1
                if first_missing_key is None:
                    first_missing_key = img_path
                continue

            assert au_vec.shape == (au_dim,), (
                f"AU dim mismatch: expected {au_dim}, got {au_vec.shape} "
                f"for {img_path}"
            )
            features_list.append(au_vec)
            labels_list.append(label)
            paths_list.append(img_path)

    features = np.stack(features_list, axis=0).astype(np.float32)  # [N, au_dim]
    labels = np.array(labels_list, dtype=np.int64)  # [N]

    coverage = len(features) / n_total_lines * 100 if n_total_lines > 0 else 0
    logger.info(
        f"Loaded {len(features)}/{n_total_lines} frames from {data_list_path} "
        f"(coverage={coverage:.1f}%, missing={n_missing})"
    )
    if n_missing > 0:
        logger.warning(
            f"  {n_missing} frames in data list have no matching NPZ key! "
            f"First missing: '{first_missing_key}'"
        )
        # Show a sample NPZ key for debugging key format mismatch
        sample_npz_key = next(iter(npz_data.keys()))
        logger.warning(f"  Sample NPZ key:     '{sample_npz_key}'")
        logger.warning(f"  First missing path:  '{first_missing_key}'")
    if coverage < 95.0:
        logger.error(
            f"  CRITICAL: NPZ coverage is only {coverage:.1f}%. "
            f"Results will be unreliable. Check key format mismatch."
        )

    # Per-class counts
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        logger.info(f"  class {cls}: {cnt} frames ({cnt / len(labels) * 100:.1f}%)")

    return features, labels, paths_list


# ---------------------------------------------------------------------------
# Video-level aggregation (matching Runner._video_level_aggregation exactly)
# ---------------------------------------------------------------------------

def compute_video_metrics(
    paths: List[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    method_name: str = "prediction",
) -> Dict:
    """Aggregate frame predictions to video-level, matching Runner protocol.

    Args:
        paths: frame image paths
        targets: [N] ground-truth labels (int)
        predictions: [N] predicted values (float, can be fractional)
        method_name: name for logging

    Returns:
        dict with mae, acc, num_videos, per_video details
    """
    video_groups: Dict[str, Dict] = defaultdict(
        lambda: {"targets": [], "preds": []}
    )
    for i, path in enumerate(paths):
        stem = PurePosixPath(path).stem
        video_id = stem.rsplit("_", 1)[0]
        video_groups[video_id]["targets"].append(float(targets[i]))
        video_groups[video_id]["preds"].append(float(predictions[i]))

    video_mae_list = []
    video_acc_list = []

    for vid, data in video_groups.items():
        targets_sorted = sorted(data["targets"])
        gt = targets_sorted[len(targets_sorted) // 2]  # median GT
        pred_mean = sum(data["preds"]) / len(data["preds"])  # mean pred

        video_mae_list.append(abs(pred_mean - gt))
        video_acc_list.append(1.0 if round(pred_mean) == gt else 0.0)

    n_videos = len(video_groups)
    mae = sum(video_mae_list) / n_videos
    acc = sum(video_acc_list) / n_videos

    logger.info(
        f"  [{method_name}] video-level: {n_videos} videos, "
        f"mae={mae:.4f}, acc={acc:.4f} ({acc * 100:.2f}%)"
    )

    return {"mae": round(mae, 6), "acc": round(acc, 6), "num_videos": n_videos}


def compute_frame_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> Dict:
    """Compute frame-level mae and acc."""
    pred_rounded = np.round(predictions).astype(np.int64)
    acc = float(np.mean(pred_rounded == targets))
    mae = float(np.mean(np.abs(predictions - targets.astype(np.float32))))
    return {"acc": round(acc, 6), "mae": round(mae, 6), "n_frames": len(targets)}


# ---------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------

def log_feature_stats(features: np.ndarray, name: str, au_dim: int) -> None:
    """Log per-dimension statistics for sanity check."""
    mean = features.mean(axis=0)  # [au_dim]
    std = features.std(axis=0)  # [au_dim]
    logger.info(f"Feature stats ({name}, {au_dim}-d):")
    for i in range(features.shape[1]):
        logger.info(
            f"  dim {i:2d}: mean={mean[i]:.4f}, std={std[i]:.4f}, "
            f"min={features[:, i].min():.4f}, max={features[:, i].max():.4f}"
        )
    # Detect constant dimensions (likely AU45/AU43 extraction bug)
    constant_dims = []
    for i in range(features.shape[1]):
        if std[i] < 1e-6:
            constant_dims.append(i)
    if constant_dims:
        logger.warning(
            f"  CONSTANT dimensions detected: {constant_dims} (std < 1e-6). "
            f"These contribute no information. If dim {features.shape[1]-1} is "
            f"constant and au_dim=17, this is likely the AU45 extraction bug — "
            f"py-feat does not output AU45 (only up to AU43). "
            f"Re-extract with the fixed extract_au_pyfeat.py."
        )
    n_low_var = int(np.sum(std < 0.01))
    if n_low_var > 0 and n_low_var != len(constant_dims):
        logger.warning(
            f"  {n_low_var} dimensions have std < 0.01 (near-constant, "
            f"may carry minimal signal)"
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class StandardScaler:
    """Simple z-score scaler (avoid sklearn dependency for this)."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)  # [D]
        self.std_ = X.std(axis=0)  # [D]
        self.std_[self.std_ < 1e-8] = 1.0  # avoid div-by-zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None
        return (X - self.mean_) / self.std_


# ---------------------------------------------------------------------------
# SVM baselines
# ---------------------------------------------------------------------------

def run_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    paths_test: List[str],
) -> Dict:
    """Run SVM baselines (RBF + Linear). Returns results dict."""
    from sklearn.svm import SVC, LinearSVC

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # --- RBF SVM ---
    logger.info("Training SVM (RBF kernel)...")
    svm_rbf = SVC(
        kernel="rbf",
        class_weight="balanced",
        C=1.0,
        gamma="scale",
        random_state=42,
    )
    svm_rbf.fit(X_train_s, y_train)

    preds_rbf = svm_rbf.predict(X_test_s).astype(np.float32)  # [N]
    frame_rbf = compute_frame_metrics(y_test, preds_rbf)
    video_rbf = compute_video_metrics(paths_test, y_test, preds_rbf, "SVM-RBF")

    # Per-class accuracy
    per_class_rbf = {}
    for cls in sorted(np.unique(y_test)):
        mask = y_test == cls
        cls_acc = float(np.mean(preds_rbf[mask] == cls))
        per_class_rbf[f"class_{cls}"] = round(cls_acc, 4)
    logger.info(f"  SVM-RBF per-class acc: {per_class_rbf}")

    results["svm_rbf"] = {
        "frame": frame_rbf,
        "video": video_rbf,
        "per_class_acc": per_class_rbf,
    }

    # --- Linear SVM ---
    logger.info("Training SVM (Linear kernel)...")
    svm_lin = LinearSVC(
        class_weight="balanced",
        C=1.0,
        max_iter=5000,
        random_state=42,
    )
    svm_lin.fit(X_train_s, y_train)

    preds_lin = svm_lin.predict(X_test_s).astype(np.float32)  # [N]
    frame_lin = compute_frame_metrics(y_test, preds_lin)
    video_lin = compute_video_metrics(paths_test, y_test, preds_lin, "SVM-Linear")

    per_class_lin = {}
    for cls in sorted(np.unique(y_test)):
        mask = y_test == cls
        cls_acc = float(np.mean(preds_lin[mask] == cls))
        per_class_lin[f"class_{cls}"] = round(cls_acc, 4)
    logger.info(f"  SVM-Linear per-class acc: {per_class_lin}")

    results["svm_linear"] = {
        "frame": frame_lin,
        "video": video_lin,
        "per_class_acc": per_class_lin,
    }

    return results


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------

class AUClassifier(nn.Module):
    """Minimal MLP for AU-only classification."""

    def __init__(self, au_dim: int, num_classes: int, hidden_dims: Tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = au_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, au_dim] -> logits: [B, num_classes]"""
        return self.net(x)


def run_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    paths_test: List[str],
    au_dim: int,
    num_classes: int,
    seed: int,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 20,
) -> Dict:
    """Train MLP and evaluate. Returns results dict."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"MLP training on {device}, epochs={epochs}, bs={batch_size}, lr={lr}")

    # Normalize
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Split 10% of training data for validation (early stopping)
    n_total = len(X_train_s)
    indices = np.random.permutation(n_total)
    n_val = max(1, int(n_total * 0.1))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_tr = torch.tensor(X_train_s[train_idx], dtype=torch.float32)  # [N_tr, au_dim]
    y_tr = torch.tensor(y_train[train_idx], dtype=torch.long)  # [N_tr]
    X_va = torch.tensor(X_train_s[val_idx], dtype=torch.float32)  # [N_va, au_dim]
    y_va = torch.tensor(y_train[val_idx], dtype=torch.long)  # [N_va]
    X_te = torch.tensor(X_test_s, dtype=torch.float32)  # [N_te, au_dim]
    y_te = torch.tensor(y_test, dtype=torch.long)  # [N_te]

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = AUClassifier(au_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class-weighted CE loss
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    ce_weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=ce_weight)

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)  # [B, au_dim], [B]
            logits = model(xb)  # [B, num_classes]
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += xb.size(0)

        scheduler.step()
        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va.to(device))  # [N_va, num_classes]
            val_preds = val_logits.argmax(dim=1).cpu()
            val_acc = float((val_preds == y_va).float().mean())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info(
                f"  epoch {epoch:3d}/{epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_acc={val_acc:.4f}, best_val={best_val_acc:.4f} (ep{best_epoch})"
            )

        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch}, best={best_epoch}")
            break

    # --- Test with best model ---
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_logits = model(X_te.to(device))  # [N_te, num_classes]
        test_probs = F.softmax(test_logits, dim=1).cpu().numpy()  # [N_te, num_classes]

    # Prediction strategies (matching OrdinalCLIP)
    pred_max = test_logits.argmax(dim=1).cpu().numpy().astype(np.float32)  # [N_te]
    rank_values = np.arange(num_classes, dtype=np.float32)  # [num_classes]
    pred_exp = (test_probs * rank_values).sum(axis=1)  # [N_te]

    y_test_np = y_test.astype(np.int64)

    # Frame-level
    frame_max = compute_frame_metrics(y_test_np, pred_max)
    frame_exp = compute_frame_metrics(y_test_np, pred_exp)
    logger.info(f"  MLP frame-level: acc_max={frame_max['acc']:.4f}, acc_exp={frame_exp['acc']:.4f}")

    # Video-level
    video_max = compute_video_metrics(paths_test, y_test_np, pred_max, "MLP-max")
    video_exp = compute_video_metrics(paths_test, y_test_np, pred_exp, "MLP-exp")

    # Per-class accuracy (argmax)
    per_class = {}
    for cls in range(num_classes):
        mask = y_test_np == cls
        if mask.sum() > 0:
            cls_acc = float(np.mean(np.round(pred_max[mask]) == cls))
            per_class[f"class_{cls}"] = round(cls_acc, 4)
    logger.info(f"  MLP per-class acc (max): {per_class}")

    return {
        "mlp": {
            "frame": {
                "acc_max": frame_max["acc"],
                "acc_exp": frame_exp["acc"],
                "mae_max": frame_max["mae"],
                "mae_exp": frame_exp["mae"],
                "n_frames": frame_max["n_frames"],
            },
            "video": {
                "acc_max": video_max["acc"],
                "acc_exp": video_exp["acc"],
                "mae_max": video_max["mae"],
                "mae_exp": video_exp["mae"],
                "num_videos": video_max["num_videos"],
            },
            "per_class_acc": per_class,
            "best_epoch": best_epoch,
            "best_val_acc": round(best_val_acc, 6),
            "train_acc_final": round(train_acc, 6),
        }
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def print_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, name: str
) -> None:
    """Print a confusion matrix to the log."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pr in zip(y_true, y_pred):
        cm[int(gt), int(pr)] += 1

    logger.info(f"\n{'='*50}")
    logger.info(f"Confusion Matrix ({name}):")
    header = "GT\\Pred | " + " ".join(f"{i:5d}" for i in range(num_classes))
    logger.info(header)
    logger.info("-" * len(header))
    for i in range(num_classes):
        row = f"rank {i}  | " + " ".join(f"{cm[i, j]:5d}" for j in range(num_classes))
        total = cm[i].sum()
        acc = cm[i, i] / total * 100 if total > 0 else 0
        row += f"  ({acc:.1f}%)"
        logger.info(row)
    logger.info(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AU-Only Baseline for BioVid Pain Classification")
    parser.add_argument("--au_npz", type=str, default="data/biovid/au_features_all17.npz")
    parser.add_argument("--au_dim", type=int, default=17, choices=[8, 17])
    parser.add_argument("--train_list", type=str, default="data/biovid/train_skip2.txt")
    parser.add_argument("--test_list", type=str, default="data/biovid/test_skip2.txt")
    parser.add_argument("--output_dir", type=str, default="results/au-only-baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--skip_svm", action="store_true")
    parser.add_argument("--skip_mlp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AU-Only Baseline Experiment")
    logger.info(f"  au_npz:      {args.au_npz}")
    logger.info(f"  au_dim:      {args.au_dim}")
    logger.info(f"  train_list:  {args.train_list}")
    logger.info(f"  test_list:   {args.test_list}")
    logger.info(f"  output_dir:  {args.output_dir}")
    logger.info(f"  seed:        {args.seed}")
    logger.info(f"  num_classes: {args.num_classes}")
    logger.info("=" * 60)

    # Load data
    X_train, y_train, paths_train = load_au_dataset(args.au_npz, args.train_list, args.au_dim)
    X_test, y_test, paths_test = load_au_dataset(args.au_npz, args.test_list, args.au_dim)

    log_feature_stats(X_train, "train", args.au_dim)

    all_results: Dict = {
        "config": {
            "au_npz": args.au_npz,
            "au_dim": args.au_dim,
            "train_list": args.train_list,
            "test_list": args.test_list,
            "seed": args.seed,
            "num_classes": args.num_classes,
            "n_train_frames": len(X_train),
            "n_test_frames": len(X_test),
        }
    }

    # --- SVM ---
    if not args.skip_svm:
        logger.info("\n" + "=" * 60)
        logger.info("SVM Baselines")
        logger.info("=" * 60)
        svm_results = run_svm(X_train, y_train, X_test, y_test, paths_test)
        all_results.update(svm_results)

        # Print confusion matrices for SVM
        from sklearn.svm import SVC, LinearSVC

        scaler = StandardScaler().fit(X_train)
        X_test_s = scaler.transform(X_test)

        svm_rbf = SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale", random_state=42)
        svm_rbf.fit(scaler.transform(X_train), y_train)
        print_confusion_matrix(y_test, svm_rbf.predict(X_test_s), args.num_classes, "SVM-RBF")

    # --- MLP ---
    if not args.skip_mlp:
        logger.info("\n" + "=" * 60)
        logger.info("MLP Baseline")
        logger.info("=" * 60)
        mlp_results = run_mlp(
            X_train, y_train, X_test, y_test, paths_test,
            au_dim=args.au_dim,
            num_classes=args.num_classes,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        all_results.update(mlp_results)

    # --- Save results ---
    out_path = os.path.join(args.output_dir, "au_baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"AU dimension: {args.au_dim}")
    logger.info(f"Random baseline: acc=20.00%, mae~1.6")
    logger.info("")

    if "svm_rbf" in all_results:
        v = all_results["svm_rbf"]["video"]
        logger.info(f"SVM-RBF    video: acc={v['acc']*100:.2f}%, mae={v['mae']:.4f}")
    if "svm_linear" in all_results:
        v = all_results["svm_linear"]["video"]
        logger.info(f"SVM-Linear video: acc={v['acc']*100:.2f}%, mae={v['mae']:.4f}")
    if "mlp" in all_results:
        v = all_results["mlp"]["video"]
        logger.info(f"MLP-max    video: acc={v['acc_max']*100:.2f}%, mae={v['mae_max']:.4f}")
        logger.info(f"MLP-exp    video: acc={v['acc_exp']*100:.2f}%, mae={v['mae_exp']:.4f}")

    logger.info("")
    random_acc = 1.0 / args.num_classes
    best_acc = 0.0
    best_method = ""
    for method_key in ["svm_rbf", "svm_linear"]:
        if method_key in all_results:
            a = all_results[method_key]["video"]["acc"]
            if a > best_acc:
                best_acc = a
                best_method = method_key
    if "mlp" in all_results:
        for sub in ["acc_max", "acc_exp"]:
            a = all_results["mlp"]["video"][sub]
            if a > best_acc:
                best_acc = a
                best_method = f"mlp_{sub}"

    if best_acc > random_acc + 0.05:
        logger.info(
            f"AU features show signal: best {best_method} = {best_acc*100:.2f}% "
            f"(>{random_acc*100:.0f}%+5%)"
        )
    elif best_acc > random_acc + 0.02:
        logger.info(
            f"AU features show weak signal: best {best_method} = {best_acc*100:.2f}% "
            f"(slightly above random {random_acc*100:.0f}%)"
        )
    else:
        logger.info(
            f"AU features show NO signal: best {best_method} = {best_acc*100:.2f}% "
            f"(~= random {random_acc*100:.0f}%)"
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()

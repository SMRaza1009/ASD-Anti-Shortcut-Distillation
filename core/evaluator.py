"""
Evaluation pipeline: clean accuracy, corruption robustness (mCE), per-corruption analysis.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
from collections import OrderedDict

from data.datasets import CIFAR100C_CORRUPTIONS, get_cifar100c_loader
from core.trainer import evaluate_model
from utils.helpers import setup_logging


# AlexNet error rates for mCE normalization (CIFAR-100-C, severity 5)
# Source: Hendrycks & Dietterich (2019) Table 3 — CIFAR-100-C AlexNet baseline
ALEXNET_ERR_CIFAR100 = {
    "gaussian_noise": 82.13, "shot_noise": 80.72, "impulse_noise": 83.11,
    "defocus_blur": 75.18, "glass_blur": 80.52, "motion_blur": 71.40,
    "zoom_blur": 76.47, "snow": 77.84, "frost": 75.31, "fog": 73.58,
    "brightness": 56.34, "contrast": 79.55, "elastic_transform": 65.72,
    "pixelate": 74.21, "jpeg_compression": 68.83,
}

# AlexNet error rates for mCE normalization (ImageNet-C, severity 5)
# Source: Hendrycks & Dietterich (2019) Table 2
ALEXNET_ERR_IMAGENET = {
    "gaussian_noise": 88.10, "shot_noise": 89.40, "impulse_noise": 92.30,
    "defocus_blur": 82.60,   "glass_blur": 90.20,  "motion_blur": 79.20,
    "zoom_blur": 86.90,      "snow": 87.20,         "frost": 83.10,
    "fog": 81.90,            "brightness": 56.50,   "contrast": 85.30,
    "elastic_transform": 64.60, "pixelate": 77.60,  "jpeg_compression": 77.10,
}

# TinyImageNet-C: use ImageNet AlexNet baseline (standard approximation — no published
# TinyImageNet-C AlexNet baseline exists in the original Hendrycks paper)
ALEXNET_ERR_TINYIMAGENET = ALEXNET_ERR_IMAGENET


@torch.no_grad()
def evaluate_clean(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate on clean test set."""
    acc = evaluate_model(model, test_loader, device)
    return {"clean_acc": acc, "clean_err": 100.0 - acc}


def _get_corruption_loader(dataset: str, data_root: str, corruption: str,
                           severity: int, batch_size: int):
    """
    Return a DataLoader for one corruption type.
    For cifar100: uses local .npy files.
    For tinyimagenet / imagenet100: tries local first, then HuggingFace.
    """
    if dataset == "cifar100":
        from data.datasets import get_cifar100c_loader
        return get_cifar100c_loader(data_root, corruption, severity, batch_size)

    if dataset == "tinyimagenet":
        # Try local first
        local_path = os.path.join(data_root, corruption)
        if os.path.isdir(local_path):
            from data.datasets import get_tinyimagenetc_loader  # type: ignore
            return get_tinyimagenetc_loader(data_root, corruption, severity, batch_size)
        # Fall back to HuggingFace
        from data.hf_datasets import get_tinyimagenet_c_hf_loader
        return get_tinyimagenet_c_hf_loader(corruption, severity, batch_size)

    if dataset == "imagenet100":
        # Try local first
        local_path = os.path.join(data_root, corruption, str(severity))
        if os.path.isdir(local_path):
            from data.datasets import get_imagenet100c_loader
            return get_imagenet100c_loader(data_root, corruption, severity, batch_size)
        # Fall back to HuggingFace (on-the-fly generation)
        from data.hf_datasets import get_imagenet100c_hf_loader
        return get_imagenet100c_hf_loader(corruption, severity, batch_size)

    raise ValueError(f"Unknown dataset for corruption eval: {dataset}")


@torch.no_grad()
def evaluate_corruption(
    model: nn.Module,
    data_root: str = "./data/CIFAR-100-C",
    severity: int = 5,
    batch_size: int = 128,
    device: str = "cuda",
    corruptions: Optional[List[str]] = None,
    dataset: str = "cifar100",
) -> Dict[str, float]:
    """
    Evaluate corruption robustness (mCE) across 15 standard corruption types.
    For cifar100: reads local CIFAR-100-C .npy files.
    For tinyimagenet/imagenet100: falls back to HuggingFace if local data absent.
    """
    model.eval()
    if corruptions is None:
        corruptions = CIFAR100C_CORRUPTIONS

    if dataset == "imagenet100":
        alexnet_table = ALEXNET_ERR_IMAGENET
    elif dataset == "tinyimagenet":
        alexnet_table = ALEXNET_ERR_TINYIMAGENET
    else:
        alexnet_table = ALEXNET_ERR_CIFAR100

    results = OrderedDict()

    for corruption in corruptions:
        try:
            loader = _get_corruption_loader(dataset, data_root, corruption, severity, batch_size)
            acc = evaluate_model(model, loader, device)
            err = 100.0 - acc
            alexnet_err = alexnet_table.get(corruption, err)
            ce = err / alexnet_err * 100.0 if alexnet_err > 0 else err
            results[corruption] = {"acc": acc, "err": err, "ce": ce}
        except (FileNotFoundError, Exception) as e:
            results[corruption] = {"acc": 0, "err": 100, "ce": 100, "error": str(e)}

    ce_values = [v["ce"] for v in results.values() if "error" not in v]
    mce = np.mean(ce_values) if ce_values else 0
    err_values = [v["err"] for v in results.values() if "error" not in v]
    mean_err = np.mean(err_values) if err_values else 0

    return {
        "per_corruption": results,
        "mCE": mce,
        "mean_err": mean_err,
        "severity": severity,
    }


@torch.no_grad()
def compute_ece(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE) with equal-width confidence bins.
    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
    """
    model.eval()
    confidences, correct = [], []
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        logits, _ = model(images)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        confidences.append(conf.cpu())
        correct.append(pred.eq(targets).cpu())

    confidences = torch.cat(confidences).numpy()
    correct = torch.cat(correct).numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    corruption_data_root: str = "./data/CIFAR-100-C",
    device: str = "cuda",
    output_path: Optional[str] = None,
    dataset: str = "cifar100",
) -> Dict:
    """Run complete evaluation: clean accuracy + ECE + all corruptions."""
    clean_results = evaluate_clean(model, test_loader, device)
    ece = compute_ece(model, test_loader, device)
    clean_results["ece"] = ece
    corruption_results = evaluate_corruption(model, corruption_data_root, device=device, dataset=dataset)

    results = {
        "clean": clean_results,
        "corruption": corruption_results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def print_corruption_table(results: Dict):
    """Pretty-print corruption evaluation results."""
    print("\n" + "=" * 70)
    print(f"{'Corruption':<25} {'Acc (%)':<10} {'Error (%)':<12} {'CE':<10}")
    print("-" * 70)

    for name, vals in results["per_corruption"].items():
        if "error" in vals:
            print(f"{name:<25} {'N/A':<10} {'N/A':<12} {'N/A':<10}")
        else:
            print(f"{name:<25} {vals['acc']:<10.1f} {vals['err']:<12.1f} {vals['ce']:<10.1f}")

    print("-" * 70)
    print(f"{'Mean':<25} {'':<10} {results['mean_err']:<12.1f} {results['mCE']:<10.1f}")
    print("=" * 70)


def print_full_results(results: Dict):
    """Print clean accuracy, ECE, and corruption table."""
    clean = results.get("clean", {})
    print(f"\nClean Acc: {clean.get('clean_acc', 0):.2f}% | "
          f"Clean Err: {clean.get('clean_err', 0):.2f}% | "
          f"ECE: {clean.get('ece', float('nan')):.4f}")
    print_corruption_table(results["corruption"])

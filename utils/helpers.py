"""
Utility functions: logging, checkpointing, metrics, reproducibility.
"""
import os
import json
import random
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
from collections import defaultdict


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(output_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class AverageMeter:
    """Computes and stores average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """Track multiple metrics across an epoch."""

    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v, n)

    def avg(self, key: str) -> float:
        return self.meters[key].avg

    def summary(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}

    def reset(self):
        for m in self.meters.values():
            m.reset()


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes accuracy for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def save_checkpoint(
    state: dict,
    filepath: str,
    is_best: bool = False,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        best_path = filepath.replace(".pth", "_best.pth")
        torch.save(state, best_path)


def load_checkpoint(filepath: str, model: nn.Module, optimizer=None, device="cuda"):
    """Load model checkpoint."""
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_acc", 0)


def save_results(results: dict, filepath: str):
    """Save results dictionary as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self) -> str:
        secs = int(time.time() - self.start_time)
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

"""
Dataset loading for ASD experiments.
Supports CIFAR-100, CIFAR-100-C, and ImageNet-100.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T
from typing import Tuple, Optional, List


# ═══════════════════════════════════════════════════════════
#  Transforms
# ═══════════════════════════════════════════════════════════

def get_cifar100_transforms() -> Tuple[T.Compose, T.Compose]:
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    return train_transform, test_transform


def get_imagenet100_transforms() -> Tuple[T.Compose, T.Compose]:
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return train_transform, test_transform


# ═══════════════════════════════════════════════════════════
#  CIFAR-100
# ═══════════════════════════════════════════════════════════

def get_cifar100_loaders(
    data_root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = get_cifar100_transforms()

    train_set = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════
#  CIFAR-100-C (Corruption Benchmark)
# ═══════════════════════════════════════════════════════════

CIFAR100C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


class CIFAR100C(Dataset):
    """CIFAR-100-C corruption benchmark dataset."""

    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int = 5,
        transform=None,
    ):
        assert corruption in CIFAR100C_CORRUPTIONS, f"Unknown: {corruption}"
        assert 1 <= severity <= 5

        data_path = os.path.join(root, f"{corruption}.npy")
        label_path = os.path.join(root, "labels.npy")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"CIFAR-100-C not found at {root}. "
                "Download from: https://zenodo.org/records/3555552"
            )

        # Each corruption file has 50000 images (10000 per severity)
        all_data = np.load(data_path)
        all_labels = np.load(label_path)

        # Select severity level (0-indexed: severity 1 = indices 0:10000)
        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = all_data[start:end]
        self.labels = all_labels[start:end].astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]  # uint8 numpy array (H, W, C)
        label = self.labels[idx]

        if self.transform:
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label


def get_cifar100c_loader(
    data_root: str = "./data/CIFAR-100-C",
    corruption: str = "gaussian_noise",
    severity: int = 5,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    _, test_tf = get_cifar100_transforms()
    dataset = CIFAR100C(data_root, corruption, severity, transform=test_tf)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ═══════════════════════════════════════════════════════════
#  ImageNet-100 (subset of ImageNet)
# ═══════════════════════════════════════════════════════════

def get_imagenet100_class_indices(full_dataset, num_classes: int = 100) -> List[int]:
    """Get indices for first num_classes of ImageNet."""
    targets = np.array([s[1] for s in full_dataset.samples])
    selected_classes = sorted(list(set(targets)))[:num_classes]
    indices = [i for i, t in enumerate(targets) if t in selected_classes]
    return indices, selected_classes


def get_imagenet100_loaders(
    data_root: str = "/data/imagenet",
    batch_size: int = 64,
    num_workers: int = 4,
    num_classes: int = 100,
) -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = get_imagenet100_transforms()

    train_set = torchvision.datasets.ImageFolder(
        os.path.join(data_root, "train"), transform=train_tf
    )
    test_set = torchvision.datasets.ImageFolder(
        os.path.join(data_root, "val"), transform=test_tf
    )

    # Subset to first 100 classes
    train_indices, _ = get_imagenet100_class_indices(train_set, num_classes)
    test_indices, _ = get_imagenet100_class_indices(test_set, num_classes)

    train_subset = Subset(train_set, train_indices)
    test_subset = Subset(test_set, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════
#  ImageNet-100-C (Corruption Benchmark)
# ═══════════════════════════════════════════════════════════

IMAGENET100C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def get_imagenet100c_loader(
    data_root: str = "./data/ImageNet-100-C",
    corruption: str = "gaussian_noise",
    severity: int = 5,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """
    Load ImageNet-100-C corruption benchmark.

    Expects directory structure:
        data_root/{corruption}/severity_{severity}/  (ImageFolder-compatible)
    or equivalently:
        data_root/{corruption}/{severity}/

    Download ImageNet-C from: https://zenodo.org/records/2235448
    Then subset to the same 100 classes used for ImageNet-100 training.
    """
    assert corruption in IMAGENET100C_CORRUPTIONS, f"Unknown corruption: {corruption}"
    assert 1 <= severity <= 5

    # Try both common directory layouts
    corruption_dir = os.path.join(data_root, corruption, str(severity))
    if not os.path.isdir(corruption_dir):
        raise FileNotFoundError(
            f"ImageNet-100-C not found at {corruption_dir}. "
            "Download from: https://zenodo.org/records/2235448 and extract under "
            f"{data_root}/{{corruption}}/{{severity}}/"
        )

    _, test_tf = get_imagenet100_transforms()
    dataset = torchvision.datasets.ImageFolder(corruption_dir, transform=test_tf)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ═══════════════════════════════════════════════════════════
#  Unified factory
# ═══════════════════════════════════════════════════════════

def get_tinyimagenet_transforms() -> Tuple[T.Compose, T.Compose]:
    """Transforms for 64×64 TinyImageNet."""
    mean, std = (0.480, 0.448, 0.398), (0.277, 0.269, 0.282)
    train_transform = T.Compose([
        T.Resize(72),
        T.RandomCrop(64),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_transform = T.Compose([
        T.Resize(64),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_transform, test_transform


def get_dataloaders(
    dataset: str,
    data_root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    hf_streaming: bool = False,
):
    """
    Unified data loader factory.

    For 'tinyimagenet' and 'imagenet100': loads from HuggingFace Hub if local
    data is not found at data_root. Requires `pip install datasets`.

    Args:
        hf_streaming: For HF-backed datasets, stream online (True) or cache locally (False).
    """
    if dataset == "cifar100":
        return get_cifar100_loaders(data_root, batch_size, num_workers)

    elif dataset == "tinyimagenet":
        # Try local ImageFolder first
        train_dir = os.path.join(data_root, "tiny-imagenet-200", "train")
        if os.path.isdir(train_dir):
            tf_train, tf_test = get_tinyimagenet_transforms()
            val_dir = os.path.join(data_root, "tiny-imagenet-200", "val")
            train_set = torchvision.datasets.ImageFolder(train_dir, transform=tf_train)
            val_set = torchvision.datasets.ImageFolder(val_dir, transform=tf_test)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
            return train_loader, val_loader
        else:
            # Fall back to HuggingFace
            from data.hf_datasets import get_tinyimagenet_hf_loaders
            return get_tinyimagenet_hf_loaders(batch_size, num_workers, streaming=hf_streaming)

    elif dataset == "imagenet100":
        # Try local ImageFolder first (standard layout: data_root/train, data_root/val)
        local_train = os.path.join(data_root, "train")
        if os.path.isdir(local_train):
            return get_imagenet100_loaders(data_root, batch_size, num_workers)
        else:
            # Fall back to HuggingFace Hub
            from data.hf_datasets import get_imagenet100_hf_loaders
            return get_imagenet100_hf_loaders(batch_size, num_workers, streaming=hf_streaming)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

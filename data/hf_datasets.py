"""
HuggingFace online dataset loaders for ASD experiments.
Provides streaming/cached access without manual downloads.

Supported datasets (all free on HuggingFace Hub):
  - ImageNet-100   : clane9/imagenet-100   (100-class ImageNet subset, 224×224)
  - TinyImageNet   : zh-plus/tiny-imagenet (200 classes, 64×64)
  - TinyImageNet-C : randall-lab/tiny-imagenet-c (corruption benchmark)

Usage:
    pip install datasets  # one-time
    from data.hf_datasets import get_imagenet100_hf_loaders, get_tinyimagenet_hf_loaders

Streaming vs. cached:
  - streaming=True : no disk writes, iterates online (slower per-epoch but no storage needed)
  - streaming=False: downloads and caches locally in ~/.cache/huggingface/ (~12 GB imagenet100,
                     ~500 MB tinyimagenet). Subsequent runs read from cache. Recommended for
                     multi-epoch training.
"""
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
import torchvision.transforms as T
from typing import Tuple, Optional, Dict
import numpy as np


# ═══════════════════════════════════════════════════════════
#  Transforms
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


def get_imagenet_transforms() -> Tuple[T.Compose, T.Compose]:
    """Standard ImageNet transforms (224×224)."""
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_transform, test_transform


# ═══════════════════════════════════════════════════════════
#  HuggingFace Dataset Wrappers
# ═══════════════════════════════════════════════════════════

class HFIterableDataset(IterableDataset):
    """Wraps a streaming HuggingFace dataset as a PyTorch IterableDataset."""

    def __init__(self, hf_dataset, transform=None, image_key="image", label_key="label"):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __iter__(self):
        for sample in self.hf_dataset:
            img = sample[self.image_key]
            label = sample[self.label_key]

            # Ensure PIL Image
            if not hasattr(img, 'convert'):
                from PIL import Image
                import io
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img)).convert("RGB")
                else:
                    img = Image.fromarray(np.array(img)).convert("RGB")
            else:
                img = img.convert("RGB")

            if self.transform:
                img = self.transform(img)

            if isinstance(label, str):
                label = int(label)

            yield img, label


class HFMapDataset(Dataset):
    """Wraps a non-streaming (cached) HuggingFace dataset as a PyTorch Dataset."""

    def __init__(self, hf_dataset, transform=None, image_key="image", label_key="label"):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

        # Build label→int mapping if labels are strings
        sample_label = hf_dataset[0][label_key]
        if isinstance(sample_label, str):
            unique_labels = sorted(set(hf_dataset[label_key]))
            self.label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            self.label_map = None

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        img = sample[self.image_key]
        label = sample[self.label_key]

        if not hasattr(img, 'convert'):
            from PIL import Image
            import io
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            else:
                img = Image.fromarray(np.array(img)).convert("RGB")
        else:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.label_map is not None:
            label = self.label_map[label]
        else:
            label = int(label)

        return img, label


# ═══════════════════════════════════════════════════════════
#  TinyImageNet loaders (zh-plus/tiny-imagenet, 200 classes)
# ═══════════════════════════════════════════════════════════

def get_tinyimagenet_hf_loaders(
    batch_size: int = 64,
    num_workers: int = 4,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load TinyImageNet from HuggingFace Hub (no local download required).

    Dataset: zh-plus/tiny-imagenet
    Classes: 200, Image size: 64×64
    Train: 100,000 images | Val: 10,000 images

    Args:
        streaming: If True, streams online (no disk needed). If False, caches in
                   ~/.cache/huggingface/ (~500 MB). Use False for multi-epoch training.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install HuggingFace datasets: pip install datasets")

    train_tf, test_tf = get_tinyimagenet_transforms()

    train_hf = load_dataset(
        "zh-plus/tiny-imagenet",
        split="train",
        streaming=streaming,
        cache_dir=cache_dir,
    )
    val_hf = load_dataset(
        "zh-plus/tiny-imagenet",
        split="valid",
        streaming=streaming,
        cache_dir=cache_dir,
    )

    if streaming:
        train_set = HFIterableDataset(train_hf.shuffle(seed=42, buffer_size=10000),
                                      transform=train_tf)
        val_set = HFIterableDataset(val_hf, transform=test_tf)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size * 2, num_workers=num_workers,
                                pin_memory=True)
    else:
        train_set = HFMapDataset(train_hf, transform=train_tf)
        val_set = HFMapDataset(val_hf, transform=test_tf)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════
#  ImageNet-100 loaders (clane9/imagenet-100, 100 classes)
# ═══════════════════════════════════════════════════════════

def get_imagenet100_hf_loaders(
    batch_size: int = 64,
    num_workers: int = 4,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load ImageNet-100 from HuggingFace Hub (no local download required).

    Dataset: clane9/imagenet-100
    Classes: 100 (fixed subset of ImageNet-1K), Image size: variable → resized to 224
    Train: 126,689 images | Val: 5,000 images

    Args:
        streaming: If True, streams online (no disk needed). If False, caches locally
                   (~12 GB). Use streaming=True to avoid disk usage.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install HuggingFace datasets: pip install datasets")

    train_tf, test_tf = get_imagenet_transforms()

    train_hf = load_dataset(
        "clane9/imagenet-100",
        split="train",
        streaming=streaming,
        cache_dir=cache_dir,
    )
    val_hf = load_dataset(
        "clane9/imagenet-100",
        split="validation",
        streaming=streaming,
        cache_dir=cache_dir,
    )

    if streaming:
        train_set = HFIterableDataset(train_hf.shuffle(seed=42, buffer_size=2000),
                                      transform=train_tf)
        val_set = HFIterableDataset(val_hf, transform=test_tf)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size * 2, num_workers=num_workers,
                                pin_memory=True)
    else:
        train_set = HFMapDataset(train_hf, transform=train_tf)
        val_set = HFMapDataset(val_hf, transform=test_tf)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════
#  TinyImageNet-C loaders (corruption benchmark)
# ═══════════════════════════════════════════════════════════

TINYIMAGENET_C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

IMAGENET100_C_CORRUPTIONS = TINYIMAGENET_C_CORRUPTIONS  # same 15 types


def get_tinyimagenet_c_hf_loader(
    corruption: str,
    severity: int = 5,
    batch_size: int = 128,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """
    Load TinyImageNet-C from HuggingFace (randall-lab/tiny-imagenet-c).
    Falls back to on-the-fly generation via imagecorruptions if HF dataset
    structure doesn't match expected splits.

    Args:
        corruption: One of TINYIMAGENET_C_CORRUPTIONS.
        severity:   1–5 (5 = most severe).
    """
    assert corruption in TINYIMAGENET_C_CORRUPTIONS, f"Unknown corruption: {corruption}"
    assert 1 <= severity <= 5

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    _, test_tf = get_tinyimagenet_transforms()

    # Try multiple split naming conventions used by randall-lab/tiny-imagenet-c
    hf_ds = None
    errors = []
    for split_name in [
        f"{corruption}_severity_{severity}",
        f"{corruption}_{severity}",
        f"test",  # some configs use a single test split
    ]:
        try:
            candidate = load_dataset(
                "randall-lab/tiny-imagenet-c",
                name=corruption,
                split=split_name,
                cache_dir=cache_dir,
            )
            # Filter to the right severity if the split contains all severities
            if "severity" in candidate.column_names:
                candidate = candidate.filter(lambda x: x["severity"] == severity)
            hf_ds = candidate
            break
        except Exception as e:
            errors.append(str(e))

    if hf_ds is None:
        # Fall back: generate on-the-fly from clean TinyImageNet val
        return _get_tinyimagenet_c_onthefly_loader(
            corruption, severity, batch_size, num_workers, cache_dir
        )

    dataset = HFMapDataset(hf_ds, transform=test_tf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def _get_tinyimagenet_c_onthefly_loader(
    corruption: str,
    severity: int,
    batch_size: int,
    num_workers: int,
    cache_dir: Optional[str],
) -> DataLoader:
    """Generate TinyImageNet-C on-the-fly from clean val + imagecorruptions."""
    try:
        from imagecorruptions import corrupt as apply_corrupt
    except ImportError:
        raise ImportError("pip install imagecorruptions")
    from datasets import load_dataset
    from PIL import Image as PILImage

    _, test_tf = get_tinyimagenet_transforms()
    # Split transform: corruption needs uint8 numpy, then we normalize
    normalize = T.Compose([T.ToTensor(),
                           T.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))])

    val_hf = load_dataset("zh-plus/tiny-imagenet", split="valid",
                          cache_dir=cache_dir)

    class TinyImageNetCDataset(Dataset):
        def __init__(self, hf_ds):
            self.hf_ds = hf_ds
            sample = hf_ds[0]
            lbl = sample.get("label", sample.get("fine_label", 0))
            if isinstance(lbl, str):
                unique = sorted(set(str(hf_ds[i].get("label", 0)) for i in range(len(hf_ds))))
                self.label_map = {l: i for i, l in enumerate(unique)}
            else:
                self.label_map = None

        def __len__(self):
            return len(self.hf_ds)

        def __getitem__(self, idx):
            sample = self.hf_ds[idx]
            img = sample.get("image", sample.get("img"))
            lbl = sample.get("label", sample.get("fine_label", 0))

            if not hasattr(img, "convert"):
                img = PILImage.fromarray(np.array(img)).convert("RGB")
            else:
                img = img.convert("RGB")

            img = img.resize((64, 64), PILImage.BILINEAR)
            arr = np.array(img, dtype=np.uint8)
            try:
                arr = apply_corrupt(arr, corruption_name=corruption, severity=severity)
            except Exception:
                pass
            img_out = normalize(PILImage.fromarray(arr.astype(np.uint8)))

            if self.label_map is not None:
                lbl = self.label_map[str(lbl)]
            return img_out, int(lbl)

    dataset = TinyImageNetCDataset(val_hf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ═══════════════════════════════════════════════════════════
#  ImageNet-100-C (on-the-fly via imagecorruptions)
# ═══════════════════════════════════════════════════════════

def get_imagenet100c_hf_loader(
    corruption: str,
    severity: int = 5,
    batch_size: int = 64,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """
    ImageNet-100-C evaluation loader — no pre-built dataset required.
    Applies Hendrycks corruptions on-the-fly to the clean clane9/imagenet-100
    validation split (5000 images, 100 classes).

    Requires: pip install imagecorruptions datasets

    Args:
        corruption: One of IMAGENET100_C_CORRUPTIONS (15 standard types).
        severity:   1–5 (5 = most severe, matches Hendrycks benchmark).
    """
    assert corruption in IMAGENET100_C_CORRUPTIONS, f"Unknown corruption: {corruption}"
    assert 1 <= severity <= 5

    try:
        from imagecorruptions import corrupt as apply_corrupt
    except ImportError:
        raise ImportError("pip install imagecorruptions")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    from PIL import Image as PILImage

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_hf = load_dataset(
        "clane9/imagenet-100",
        split="validation",
        cache_dir=cache_dir,
    )

    # Build label→int map once (clane9/imagenet-100 uses WordNet synset strings)
    sample_label = val_hf[0]["label"]
    if isinstance(sample_label, str):
        unique_labels = sorted(set(val_hf["label"]))
        label_map: Dict[str, int] = {l: i for i, l in enumerate(unique_labels)}
    else:
        label_map = None

    class ImageNet100CDataset(Dataset):
        def __len__(self):
            return len(val_hf)

        def __getitem__(self, idx):
            sample = val_hf[idx]
            img = sample["image"]
            lbl = sample["label"]

            if not hasattr(img, "convert"):
                img = PILImage.fromarray(np.array(img)).convert("RGB")
            else:
                img = img.convert("RGB")

            # Resize to 224 before corruption (matches benchmark protocol)
            img = img.resize((224, 224), PILImage.BILINEAR)
            arr = np.array(img, dtype=np.uint8)

            try:
                arr = apply_corrupt(arr, corruption_name=corruption, severity=severity)
            except Exception:
                pass  # keep clean image on failure

            img_out = normalize(PILImage.fromarray(arr.astype(np.uint8)))

            if label_map is not None:
                lbl = label_map[lbl]
            return img_out, int(lbl)

    dataset = ImageNet100CDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

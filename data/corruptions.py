"""
CIFAR-100-C corruption generation.

If pre-downloaded data is not available, generate corruptions on-the-fly
using torchvision transforms or imagecorruptions library.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# Severity-scaled corruption functions for on-the-fly generation
# These approximate the standard CIFAR-100-C corruptions

def gaussian_noise(img, severity):
    """Add Gaussian noise."""
    std = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    noise = torch.randn_like(img) * std
    return (img + noise).clamp(0, 1)


def shot_noise(img, severity):
    """Add shot (Poisson) noise."""
    lam = [60, 25, 12, 5, 3][severity - 1]
    noisy = torch.poisson(img * lam) / lam
    return noisy.clamp(0, 1)


def impulse_noise(img, severity):
    """Add salt-and-pepper noise."""
    prob = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    mask = torch.rand_like(img)
    img = img.clone()
    img[mask < prob / 2] = 0.0
    img[mask > 1 - prob / 2] = 1.0
    return img


def gaussian_blur(img, severity):
    """Apply Gaussian blur."""
    sigma = [1, 2, 3, 4, 6][severity - 1]
    kernel = int(2 * round(2 * sigma) + 1)
    return TF.gaussian_blur(img.unsqueeze(0), kernel_size=kernel, sigma=sigma).squeeze(0)


def contrast(img, severity):
    """Reduce contrast."""
    factor = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    mean = img.mean()
    return ((img - mean) * factor + mean).clamp(0, 1)


def brightness(img, severity):
    """Adjust brightness."""
    delta = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    return (img + delta).clamp(0, 1)


def fog(img, severity):
    """Simulate fog."""
    fog_density = [0.3, 0.5, 0.6, 0.75, 0.9][severity - 1]
    return (img * (1 - fog_density) + fog_density).clamp(0, 1)


def pixelate(img, severity):
    """Pixelate the image."""
    scale = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    c, h, w = img.shape
    small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
    img_small = TF.resize(img.unsqueeze(0), [small_h, small_w], antialias=True)
    return TF.resize(img_small, [h, w], interpolation=T.InterpolationMode.NEAREST).squeeze(0)


ONLINE_CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": gaussian_blur,
    "contrast": contrast,
    "brightness": brightness,
    "fog": fog,
    "pixelate": pixelate,
}


class OnlineCIFAR100C(Dataset):
    """
    Generate CIFAR-100 corruptions on-the-fly.
    Use when pre-downloaded CIFAR-100-C is not available.
    """

    def __init__(self, base_dataset, corruption: str, severity: int = 5, normalize_transform=None):
        self.base = base_dataset
        self.corruption_fn = ONLINE_CORRUPTIONS.get(corruption)
        if self.corruption_fn is None:
            raise ValueError(f"Online corruption '{corruption}' not implemented. "
                           f"Available: {list(ONLINE_CORRUPTIONS.keys())}")
        self.severity = severity
        self.normalize = normalize_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)
        elif isinstance(img, torch.Tensor) and self.normalize is None:
            pass
        img = self.corruption_fn(img, self.severity)
        if self.normalize:
            img = self.normalize(img)
        return img, label


def download_cifar100c(root: str = "./data/CIFAR-100-C"):
    """Print instructions for downloading CIFAR-100-C."""
    if os.path.exists(os.path.join(root, "gaussian_noise.npy")):
        print(f"CIFAR-100-C already exists at {root}")
        return True

    print(f"""
    CIFAR-100-C not found at {root}.

    Download from: https://zenodo.org/records/3555552
    
    Direct download:
        wget https://zenodo.org/records/3555552/files/CIFAR-100-C.tar
        tar -xf CIFAR-100-C.tar -C {os.path.dirname(root)}

    Expected structure:
        {root}/
        ├── gaussian_noise.npy
        ├── shot_noise.npy
        ├── ...
        └── labels.npy
    """)
    return False

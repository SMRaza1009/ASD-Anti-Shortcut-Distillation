"""
Centralized configuration for Anti-Shortcut Distillation experiments.
All hyperparameters, paths, and experiment settings in one place.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os


@dataclass
class DataConfig:
    dataset: str = "cifar100"
    data_root: str = "./data"
    num_workers: int = 4
    # ImageNet-100 class subset (first 100 classes)
    imagenet100_classes: int = 100

    @property
    def num_classes(self) -> int:
        return {"cifar100": 100, "cifar10": 10, "imagenet100": 100,
                "tinyimagenet": 200, "coco": 91}[self.dataset]

    @property
    def image_size(self) -> int:
        return {"cifar100": 32, "cifar10": 32, "imagenet100": 224,
                "tinyimagenet": 64, "coco": 800}[self.dataset]


@dataclass
class TeacherConfig:
    arch: str = "resnet34"
    epochs: int = 240
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 64
    lr_scheduler: str = "cosine"
    # Checkpoint saving: save every N epochs for early checkpoint selection
    save_every: int = 10
    output_dir: str = "outputs/teachers"


@dataclass
class StudentConfig:
    arch: str = "resnet18"
    epochs: int = 240
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 64
    lr_scheduler: str = "cosine"
    output_dir: str = "outputs/students"


@dataclass
class ASDConfig:
    """Anti-Shortcut Distillation hyperparameters."""
    # KD loss
    kd_temperature: float = 4.0
    alpha_kd: float = 1.0

    # Temporal Contrastive Loss (L_TC)
    alpha_tc: float = 0.8           # reduced from 1.0: in-batch negatives make TC stronger
    tc_temperature: float = 0.07    # changed from 0.1: matches CRD temperature for in-batch negatives
    tc_memory_bank_size: int = 4096  # Fix B: cross-batch negatives (0 = in-batch only)

    # Shortcut Suppression Loss (L_SS)
    alpha_ss: float = 1.0           # increased from 0.5
    ss_margin: float = 0.1          # 0.1 ≈ E[||proj||_2] for random unit vector (k=4, D=512)
    ss_k_dims: int = 4              # PCA shortcut subspace dimensionality

    # Early checkpoint selection
    early_epoch_ratio: float = 0.20  # changed from 0.10: shortcuts more developed at 20% of training
    early_epoch: Optional[int] = None  # Override: exact epoch number

    # Feature extraction
    feature_layer: str = "penultimate"  # 'penultimate' or 'last_conv'

    # Warmup: gradually introduce L_TC and L_SS
    warmup_epochs: int = 20  # changed from 5: matches DKD warmup, prevents disrupting early student learning


@dataclass
class EvalConfig:
    corruption_severities: List[int] = field(default_factory=lambda: [5])
    corruption_types: List[str] = field(default_factory=lambda: [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    ])
    batch_size: int = 128


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    asd: ASDConfig = field(default_factory=ASDConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Training method
    method: str = "asd"  # 'vanilla_kd', 'asd', 'asd_tc_only', 'asd_ss_only', 'ce_only',
                          # 'checkpoint_kd', 'fitnets', 'at', 'dkd', 'crd'

    # Reproducibility
    seed: int = 0
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Hardware
    device: str = "cuda"
    use_wandb: bool = False
    wandb_project: str = "anti-shortcut-distillation"

    # Output
    output_dir: str = "outputs"

    def get_teacher_ckpt_dir(self) -> str:
        return os.path.join(
            self.output_dir, "teachers",
            f"{self.teacher.arch}_{self.data.dataset}"
        )

    def get_student_ckpt_dir(self) -> str:
        return os.path.join(
            self.output_dir, "students",
            f"{self.method}_{self.teacher.arch}_{self.student.arch}_{self.data.dataset}_seed{self.seed}"
        )

    def get_early_epoch(self) -> int:
        if self.asd.early_epoch is not None:
            return self.asd.early_epoch
        return int(self.teacher.epochs * self.asd.early_epoch_ratio)


# ── Preset configurations for different T-S pairs ──
#
# Tier 1 (primary):  Core experiments + scale validation (Experiments 1–3 in report)
# Tier 2 (extended): Pairs from mdistiller/CRD benchmarks for Table 2 coverage
# Tier 3 (optional): Cross-dataset / deployment-target pairs

PAIR_CONFIGS = {
    # ── Tier 1 (primary ASD experiments) ───────────────────────────────────────
    "P1": {"teacher": "resnet34",   "student": "resnet18",     "dataset": "cifar100"},
    "P2": {"teacher": "wrn_40_2",   "student": "wrn_16_2",     "dataset": "cifar100"},
    "P3": {"teacher": "wrn_40_2",   "student": "shufflenet_v2","dataset": "cifar100"},
    "P4": {"teacher": "resnet50",   "student": "mobilenet_v2", "dataset": "imagenet100"},

    # ── Tier 2 (mdistiller/CRD benchmark, shared with SDMD Table 1) ────────────
    "P5": {"teacher": "resnet32x4", "student": "resnet8x4",    "dataset": "cifar100"},
    "P6": {"teacher": "wrn_40_2",   "student": "wrn_40_1",     "dataset": "cifar100"},
    "P7": {"teacher": "vgg13",      "student": "vgg8",         "dataset": "cifar100"},

    # ── Tier 3 (cross-dataset) ──────────────────────────────────────────────────
    "P8": {"teacher": "resnet34",   "student": "resnet18",     "dataset": "tinyimagenet"},

    # ── SDMD Table 1(a) — Homogeneous CIFAR-100 pairs ───────────────────────────
    # Direct comparison with SDMD (WACV 2026) Table 1 homogeneous setting.
    "S1": {"teacher": "resnet56",   "student": "resnet20",     "dataset": "cifar100"},
    # 64→64, standard CIFAR ResNets (He et al. 2016). SDMD: teacher=72.34%, student=69.06%.
    # P6 (wrn_40_2→wrn_40_1) already covers SDMD's second homogeneous pair.
    # P5 (resnet32x4→resnet8x4) already covers SDMD's third homogeneous pair.

    # ── SDMD Table 1(b) — Heterogeneous CIFAR-100 pairs ─────────────────────────
    # SDMD reports: ResNet-50→MobileNetV2, ResNet-32×4→ShuffleV1, ResNet-32×4→ShuffleV2
    "S2": {"teacher": "resnet50",   "student": "mobilenet_v2", "dataset": "cifar100"},
    # ResNet-50(cifar)→MobileNetV2(cifar). SDMD: 70.87%. Note: P4 is imagenet100 version.
    "S3": {"teacher": "resnet32x4", "student": "shufflenet_v1","dataset": "cifar100"},
    # 256→960, cross-arch with ShuffleNetV1. SDMD: 77.92%.
    "S4": {"teacher": "resnet32x4", "student": "shufflenet_v2","dataset": "cifar100"},
    # 256→1024, cross-arch with ShuffleNetV2. SDMD: 78.11%.
    # (P3 uses wrn_40_2 as teacher; S4 uses resnet32x4 to match SDMD exactly.)

    # ── SDMD Table 2 — ImageNet-100 scale (proxy for ImageNet-1K) ───────────────
    # SDMD uses RegNetY-160→DeiT (full ImageNet-1K, ViT). We use CNN on ImageNet-100
    # as a scale-validation proxy — note the difference in paper.
    # P4 (resnet50→mobilenet_v2, imagenet100) covers this proxy.

    # ── SDMD Tables 3–5 — COCO Object Detection ──────────────────────────────────
    # SDMD uses ViDT-Base→ViDT-Nano/Tiny/Small. We use Faster R-CNN with ResNet backbone.
    # Detection pairs require --task detection flag.
    "D1": {"teacher": "resnet50_det",  "student": "resnet18",    "dataset": "coco"},
    # ResNet-50 FPN teacher → ResNet-18 student. Comparable capacity gap to ViDT-Base→Nano.
    "D2": {"teacher": "resnet101_det", "student": "resnet50_det","dataset": "coco"},
    # ResNet-101 FPN teacher → ResNet-50 FPN student. Comparable to ViDT-Base→Small.
}

COCO_CONFIG_OVERRIDES = {
    # Detection training: AdamW, lower LR, shorter schedule
    "teacher": {"lr": 1e-4, "epochs": 50, "weight_decay": 1e-4, "batch_size": 4},
    "student": {"lr": 1e-4, "epochs": 50, "weight_decay": 1e-4, "batch_size": 4},
}

IMAGENET100_CONFIG_OVERRIDES = {
    "teacher": {"lr": 0.1, "epochs": 100, "weight_decay": 1e-4},
    "student": {"lr": 0.1, "epochs": 100, "weight_decay": 1e-4},
}

TINYIMAGENET_CONFIG_OVERRIDES = {
    "teacher": {"lr": 0.05, "epochs": 200, "weight_decay": 5e-4},
    "student": {"lr": 0.05, "epochs": 200, "weight_decay": 5e-4},
}

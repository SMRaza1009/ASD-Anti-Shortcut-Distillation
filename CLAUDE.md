# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Anti-Shortcut Distillation (ASD)** — a PyTorch research project implementing a novel knowledge distillation technique that teaches student networks what *not* to learn. Beyond standard KD (pulling students toward teacher outputs), ASD pushes students away from early-stage teacher shortcuts via temporal contrastive learning and shortcut suppression.

## Setup

```bash
pip install -r asd/requirements.txt
```

All scripts are run as Python modules from the repo root (not from inside `asd/`).

## Common Commands

```bash
# Train teacher (saves checkpoints every 10 epochs)
python -m scripts.train_teacher --arch resnet34 --dataset cifar100

# Train student with ASD
python -m scripts.train_student \
    --teacher resnet34 --student resnet18 \
    --method asd --dataset cifar100

# Evaluate on clean + corruption benchmarks
python -m scripts.evaluate \
    --checkpoint outputs/students/asd_resnet34_resnet18_cifar100_seed0/epoch_0240_best.pth \
    --arch resnet18 --dataset cifar100

# Run ablation studies
python -m scripts.run_ablations --ablation loss_components \
    --teacher resnet34 --student resnet18 --dataset cifar100

# Full experiment pipeline (multi-phase)
bash scripts/run_all.sh           # all phases
bash scripts/run_all.sh --phase 2 # only phase 2 (core experiments)
```

**Training methods** (`--method` flag): `ce_only`, `vanilla_kd`, `asd`, `asd_tc_only`, `asd_ss_only`

## Architecture

### Training Pipeline

```
Teacher Pre-training  →  saves T_early (epoch ~24) and T_final (epoch 240)
        ↓
Student Distillation  →  loads both T_early and T_final (frozen)
        ↓
Evaluation  →  clean accuracy + mCE across 15 corruption types × 5 severities
```

### Loss Function (core innovation — `asd/core/losses.py`)

```
L_total = L_CE + α_kd·L_KD + β(epoch)·α_tc·L_TC + γ(epoch)·α_ss·L_SS
```

- **L_KD**: Standard Hinton KD — temperature-scaled KL divergence between student and teacher logits
- **L_TC** (Temporal Contrastive): InfoNCE loss; T_final features as positives, T_early features as negatives
- **L_SS** (Shortcut Suppression): Hinge loss penalizing student activation along shortcut direction `Δh = h_early − h_final`
- β and γ use linear warmup over first 5 epochs

### Model Interface

All models in `asd/utils/models.py` return `(logits, features)` — features are penultimate-layer embeddings used for L_TC and L_SS. A projector head is added automatically when student/teacher feature dims differ.

### Configuration (`asd/configs/defaults.py`)

Dataclass hierarchy: `ExperimentConfig` contains `DataConfig`, `TeacherConfig`, `StudentConfig`, `ASDConfig`, `EvalConfig`. Key defaults: α_kd=1.0, α_tc=1.0, α_ss=0.5, τ_kd=4.0, τ_c=0.1, ε=-0.1.

Preset teacher→student pairs (PAIR_CONFIGS): ResNet-34→18 (CIFAR-100), WRN-40-2→16-2 (CIFAR-100), WRN-40-2→ShuffleNet-V2 (CIFAR-100), ResNet-50→MobileNet-V2 (ImageNet-100).

### Key Modules

| Module | Purpose |
|--------|---------|
| `asd/core/losses.py` | ASDLoss, L_TC, L_SS implementations |
| `asd/core/trainer.py` | `train_teacher()`, `train_student_asd()`, `train_student_vanilla_kd()` |
| `asd/core/evaluator.py` | Clean accuracy + mCE corruption evaluation |
| `asd/data/datasets.py` | CIFAR-100, CIFAR-100-C, ImageNet-100 loaders |
| `asd/utils/models.py` | ResNet, WideResNet, ShuffleNet, MobileNet variants |

### Checkpoint Naming

Teacher checkpoints: `outputs/teachers/{arch}_{dataset}_seed{n}/epoch_{N:04d}[_best].pth`  
Student checkpoints: `outputs/students/{method}_{teacher}_{student}_{dataset}_seed{n}/epoch_{N:04d}_best.pth`

T_early is selected as the checkpoint at 10% of total teacher training epochs (e.g., epoch 24 for 240-epoch training).

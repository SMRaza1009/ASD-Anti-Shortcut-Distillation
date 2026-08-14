# Anti-Shortcut Distillation (ASD)

> **Teaching Students What NOT to Learn via Temporal Contrastive Knowledge Transfer**

## 📄 Paper
- [Manuscript (PDF)](docs/BMVC_ASD_Manuscript.pdf)
- [Supplementary Material (PDF)](docs/BMVC_ASD_Supplementary.pdf)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train teacher with checkpoint saving
python -m scripts.train_teacher --arch resnet34 --dataset cifar100

# 3. Train student with ASD
python -m scripts.train_student \
    --teacher resnet34 \
    --student resnet18 \
    --method asd \
    --dataset cifar100

# 4. Evaluate robustness
python -m scripts.evaluate \
    --checkpoint outputs/asd_resnet34_resnet18/best.pth \
    --dataset cifar100c

# 5. Run full experiment suite
bash scripts/run_all.sh
```

## Project Structure

```
asd/
├── configs/
│   └── defaults.py          # All hyperparameters and experiment configs
├── core/
│   ├── losses.py            # L_TC, L_SS, and combined ASD loss
│   ├── trainer.py           # Training loop with ASD support
│   └── evaluator.py         # Clean + corruption evaluation
├── data/
│   ├── datasets.py          # CIFAR-100, ImageNet-100 loaders
│   └── corruptions.py       # CIFAR-100-C generation and loading
├── utils/
│   ├── models.py            # Model zoo (ResNet, WRN, MobileNet, ShuffleNet)
│   ├── helpers.py           # Logging, checkpointing, metrics
│   └── feature_hooks.py     # Feature extraction hooks for any model
├── analysis/
│   ├── visualize.py         # PCA of Δh, Grad-CAM, t-SNE
│   └── texture_bias.py      # Shape vs texture bias measurement
├── scripts/
│   ├── train_teacher.py     # Teacher training entry point
│   ├── train_student.py     # Student training entry point
│   ├── evaluate.py          # Evaluation entry point
│   ├── run_ablations.py     # Ablation study runner
│   └── run_all.sh           # Full experiment pipeline
└── requirements.txt
```

## Method Overview

Standard KD: Pull student → toward final teacher
ASD:         Pull student → toward final teacher
             Push student → AWAY FROM early teacher (shortcuts)

Two novel losses:
- **L_TC** (Temporal Contrastive): InfoNCE with final teacher as positive, early teacher as negative
- **L_SS** (Shortcut Suppression): Penalizes student for activating along Δh = h_early − h_final

## Supported Configurations

| Teacher      | Student       | Dataset      |
|-------------|---------------|--------------|
| ResNet-34   | ResNet-18     | CIFAR-100    |
| WRN-40-2    | WRN-16-2      | CIFAR-100    |
| WRN-40-2    | ShuffleNet-V2 | CIFAR-100    |
| ResNet-50   | MobileNet-V2  | ImageNet-100 |


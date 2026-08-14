"""
Train teacher model with periodic checkpoint saving for ASD.

Usage:
    python -m scripts.train_teacher --arch resnet34 --dataset cifar100
    python -m scripts.train_teacher --arch wrn_40_2 --dataset cifar100
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.defaults import ExperimentConfig
from utils.models import build_model
from utils.helpers import set_seed, setup_logging, count_parameters
from data.datasets import get_dataloaders
from core.trainer import train_teacher


def parse_args():
    parser = argparse.ArgumentParser(description="Train teacher with checkpoint saving")
    parser.add_argument("--arch", type=str, default="resnet34",
                        choices=["resnet34", "wrn_40_2", "wrn_40_4", "resnet32x4",
                                 "vgg13", "resnet50", "resnet56"])
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "imagenet100", "tinyimagenet"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    cfg.data.dataset = args.dataset
    cfg.data.data_root = args.data_root
    cfg.teacher.arch = args.arch
    cfg.teacher.batch_size = args.batch_size
    cfg.teacher.save_every = args.save_every
    cfg.seed = args.seed

    # Dataset-specific defaults
    if args.dataset == "imagenet100":
        cfg.teacher.epochs = args.epochs or 100
        cfg.teacher.lr = args.lr or 0.1
        cfg.teacher.weight_decay = 1e-4
    elif args.dataset == "tinyimagenet":
        cfg.teacher.epochs = args.epochs or 200
        cfg.teacher.lr = args.lr or 0.05
    else:
        cfg.teacher.epochs = args.epochs or 240
        cfg.teacher.lr = args.lr or 0.05

    output_dir = args.output_dir or cfg.get_teacher_ckpt_dir()
    set_seed(cfg.seed)

    logger = setup_logging(output_dir, "main")
    logger.info(f"Training teacher: {args.arch} on {args.dataset}")

    # Build model
    model = build_model(args.arch, cfg.data.num_classes, args.dataset)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Get data
    train_loader, test_loader = get_dataloaders(
        args.dataset, args.data_root, cfg.teacher.batch_size, cfg.data.num_workers
    )
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Train
    results = train_teacher(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=cfg.teacher.epochs,
        lr=cfg.teacher.lr,
        momentum=cfg.teacher.momentum,
        weight_decay=cfg.teacher.weight_decay,
        save_every=cfg.teacher.save_every,
        output_dir=output_dir,
        device=args.device,
    )

    logger.info(f"Done! Best accuracy: {results['best_acc']:.2f}%")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info(f"T_early candidates: epoch_{cfg.get_early_epoch():04d}.pth")


if __name__ == "__main__":
    main()

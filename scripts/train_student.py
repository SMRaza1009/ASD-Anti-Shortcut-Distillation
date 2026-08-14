"""
Train student model with ASD or baseline methods.

Usage:
    # ASD (full method)
    python -m scripts.train_student --teacher resnet34 --student resnet18 --method asd

    # Vanilla KD baseline
    python -m scripts.train_student --teacher resnet34 --student resnet18 --method vanilla_kd

    # CE-only baseline (no distillation)
    python -m scripts.train_student --student resnet18 --method ce_only

    # ASD ablation: TC loss only
    python -m scripts.train_student --teacher resnet34 --student resnet18 --method asd_tc_only

    # ASD ablation: SS loss only
    python -m scripts.train_student --teacher resnet34 --student resnet18 --method asd_ss_only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.defaults import ExperimentConfig
from utils.models import build_model, get_feature_dim
from utils.helpers import set_seed, setup_logging, load_checkpoint, save_results, count_parameters
from data.datasets import get_dataloaders
from core.losses import ASDLoss
from core.trainer import (
    train_student_asd, train_student_vanilla_kd, train_student_ce_only,
    train_student_fitnets, train_student_at, train_student_dkd, train_student_crd,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train student with ASD")
    # Architecture
    parser.add_argument("--teacher", type=str, default="resnet34")
    parser.add_argument("--student", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "imagenet100", "tinyimagenet"])

    # Method
    parser.add_argument("--method", type=str, default="asd",
                        choices=["asd", "vanilla_kd", "ce_only", "asd_tc_only", "asd_ss_only",
                                 "checkpoint_kd", "fitnets", "at", "dkd", "crd"])

    # ASD hyperparameters
    parser.add_argument("--alpha_kd", type=float, default=1.0)
    parser.add_argument("--alpha_tc", type=float, default=0.8)
    parser.add_argument("--alpha_ss", type=float, default=1.0)
    parser.add_argument("--kd_temp", type=float, default=4.0)
    parser.add_argument("--tc_temp", type=float, default=0.07)
    parser.add_argument("--ss_margin", type=float, default=0.1)
    parser.add_argument("--ss_k_dims", type=int, default=4,
                        help="PCA shortcut subspace dims for L_SS (default: 4)")
    parser.add_argument("--tc_memory_bank", type=int, default=4096,
                        help="Memory bank size for TC cross-batch negatives (0 = in-batch only)")
    parser.add_argument("--warmup_epochs", type=int, default=20)

    # Early checkpoint
    parser.add_argument("--early_epoch", type=int, default=None,
                        help="Exact epoch for T_early. If None, uses early_ratio of teacher epochs.")
    parser.add_argument("--early_ratio", type=float, default=0.2)

    # Training
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    # Paths
    parser.add_argument("--teacher_dir", type=str, default=None,
                        help="Directory with teacher checkpoints")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output_dir")

    return parser.parse_args()


def load_teacher(arch: str, ckpt_path: str, num_classes: int, dataset: str, device: str):
    """Load a frozen teacher model from checkpoint."""
    model = build_model(arch, num_classes, dataset)
    ckpt = load_checkpoint(ckpt_path, model, device=device)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    args = parse_args()
    cfg = ExperimentConfig()
    cfg.data.dataset = args.dataset
    cfg.data.data_root = args.data_root
    cfg.seed = args.seed
    cfg.method = args.method

    # Dataset-specific defaults
    if args.dataset == "imagenet100":
        default_epochs = 100
        default_lr = 0.1
        cfg.student.weight_decay = 1e-4
    else:
        default_epochs = 240
        default_lr = 0.05

    cfg.student.epochs = args.epochs or default_epochs
    cfg.student.lr = args.lr or default_lr
    cfg.student.arch = args.student
    cfg.student.batch_size = args.batch_size
    cfg.teacher.arch = args.teacher

    set_seed(cfg.seed)

    # Output directory
    output_dir = args.output_dir or os.path.join(
        cfg.output_dir, "students",
        f"{args.method}_{args.teacher}_{args.student}_{args.dataset}_seed{args.seed}"
    )
    logger = setup_logging(output_dir, "main")
    logger.info(f"Method: {args.method} | Teacher: {args.teacher} | Student: {args.student}")

    # ── Resume: find latest checkpoint ───────────────────────────────────────
    start_epoch = 0
    initial_best_acc = 0.0
    resume_ckpt_path = None

    if args.resume and os.path.isdir(output_dir):
        import glob as _glob
        import torch as _torch
        best_ckpts = sorted(_glob.glob(os.path.join(output_dir, "epoch_*_best.pth")))
        if best_ckpts:
            resume_ckpt_path = best_ckpts[-1]  # highest epoch = latest best
            raw = _torch.load(resume_ckpt_path, map_location=args.device)
            start_epoch = raw.get("epoch", 0)
            initial_best_acc = raw.get("best_acc", 0.0)
            logger.info(f"Resume from: {resume_ckpt_path} (epoch {start_epoch}, best {initial_best_acc:.2f}%)")
        else:
            logger.info("--resume set but no checkpoint found — starting fresh")

    # Data
    train_loader, test_loader = get_dataloaders(
        args.dataset, args.data_root, cfg.student.batch_size, cfg.data.num_workers
    )

    # Build student
    student = build_model(args.student, cfg.data.num_classes, args.dataset)
    logger.info(f"Student params: {count_parameters(student):,}")

    # Load resumed weights into student
    if resume_ckpt_path is not None:
        import torch as _torch
        raw = _torch.load(resume_ckpt_path, map_location=args.device)
        student.load_state_dict(raw["model_state_dict"])
        logger.info(f"Loaded student weights from epoch {start_epoch}")

    # ── CE-only training ──
    if args.method == "ce_only":
        results = train_student_ce_only(
            student=student,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=cfg.student.epochs,
            lr=cfg.student.lr,
            momentum=cfg.student.momentum,
            weight_decay=cfg.student.weight_decay,
            output_dir=output_dir,
            device=args.device,
            start_epoch=start_epoch,
            initial_best_acc=initial_best_acc,
        )
        save_results(results, os.path.join(output_dir, "results.json"))
        return

    # ── Load teacher(s) ──
    teacher_dir = args.teacher_dir or os.path.join(
        cfg.output_dir, "teachers", f"{args.teacher}_{args.dataset}"
    )

    # Find best teacher checkpoint
    if not os.path.isdir(teacher_dir):
        raise FileNotFoundError(
            f"Teacher checkpoint directory not found: '{teacher_dir}'\n"
            f"Either:\n"
            f"  (a) Train the teacher first:  python -m scripts.train_teacher --arch {args.teacher} --dataset {args.dataset}\n"
            f"  (b) Point to existing checkpoints: --teacher_dir /path/to/checkpoints"
        )

    best_ckpt = os.path.join(teacher_dir, "epoch_0240_best.pth")
    if not os.path.exists(best_ckpt):
        # Find the LAST _best.pth (highest epoch = best accuracy achieved)
        best_candidates = sorted([
            f for f in os.listdir(teacher_dir) if f.endswith("_best.pth")
        ])
        if best_candidates:
            best_ckpt = os.path.join(teacher_dir, best_candidates[-1])
        else:
            # Fall back to last epoch checkpoint
            ckpts = sorted([f for f in os.listdir(teacher_dir) if f.endswith(".pth") and "best" not in f])
            best_ckpt = os.path.join(teacher_dir, ckpts[-1]) if ckpts else None

    if best_ckpt is None or not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"No teacher checkpoint found in {teacher_dir}")

    logger.info(f"Loading T_final from: {best_ckpt}")
    teacher_final = load_teacher(args.teacher, best_ckpt, cfg.data.num_classes, args.dataset, args.device)

    # ── Vanilla KD ──
    if args.method == "vanilla_kd":
        results = train_student_vanilla_kd(
            student=student,
            teacher=teacher_final,
            train_loader=train_loader,
            test_loader=test_loader,
            alpha_kd=args.alpha_kd,
            kd_temperature=args.kd_temp,
            epochs=cfg.student.epochs,
            lr=cfg.student.lr,
            momentum=cfg.student.momentum,
            weight_decay=cfg.student.weight_decay,
            output_dir=output_dir,
            device=args.device,
            start_epoch=start_epoch,
            initial_best_acc=initial_best_acc,
        )
        save_results(results, os.path.join(output_dir, "results.json"))
        return

    # ── CheckpointKD: T_early used as a *positive* teacher (ablation baseline) ──
    # Per report Section 5.3: isolates whether gains come from negative-anchor use
    # vs. simply using a different (earlier) checkpoint as teacher.
    if args.method == "checkpoint_kd":
        teacher_epochs = cfg.teacher.epochs if args.dataset != "imagenet100" else 100
        early_epoch = args.early_epoch or int(teacher_epochs * args.early_ratio)
        early_ckpt = os.path.join(teacher_dir, f"epoch_{early_epoch:04d}.pth")
        if not os.path.exists(early_ckpt):
            available = sorted([
                int(f.split("_")[1].split(".")[0])
                for f in os.listdir(teacher_dir)
                if f.startswith("epoch_") and f.endswith(".pth") and "best" not in f
            ])
            nearest = min(available, key=lambda x: abs(x - early_epoch))
            early_ckpt = os.path.join(teacher_dir, f"epoch_{nearest:04d}.pth")
            logger.info(f"Exact epoch {early_epoch} not found, using nearest: epoch {nearest}")
        logger.info(f"CheckpointKD: using T_early as positive teacher from {early_ckpt}")
        teacher_early_pos = load_teacher(args.teacher, early_ckpt, cfg.data.num_classes, args.dataset, args.device)
        results = train_student_vanilla_kd(
            student=student,
            teacher=teacher_early_pos,
            train_loader=train_loader,
            test_loader=test_loader,
            alpha_kd=args.alpha_kd,
            kd_temperature=args.kd_temp,
            epochs=cfg.student.epochs,
            lr=cfg.student.lr,
            momentum=cfg.student.momentum,
            weight_decay=cfg.student.weight_decay,
            output_dir=output_dir,
            device=args.device,
            start_epoch=start_epoch,
            initial_best_acc=initial_best_acc,
        )
        save_results(results, os.path.join(output_dir, "results.json"))
        return

    # ── Baseline methods needing only T_final ──
    s_feat_dim = get_feature_dim(student.to(args.device), args.dataset)
    t_feat_dim = get_feature_dim(teacher_final, args.dataset)

    if args.method == "fitnets":
        results = train_student_fitnets(
            student=student, teacher=teacher_final,
            train_loader=train_loader, test_loader=test_loader,
            student_dim=s_feat_dim, teacher_dim=t_feat_dim,
            alpha_kd=args.alpha_kd, kd_temperature=args.kd_temp,
            epochs=cfg.student.epochs, lr=cfg.student.lr,
            momentum=cfg.student.momentum, weight_decay=cfg.student.weight_decay,
            output_dir=output_dir, device=args.device,
            start_epoch=start_epoch, initial_best_acc=initial_best_acc,
        )
        save_results(results, os.path.join(output_dir, "results.json"))
        return

    if args.method == "at":
        results = train_student_at(
            student=student, teacher=teacher_final,
            train_loader=train_loader, test_loader=test_loader,
            alpha_kd=args.alpha_kd, kd_temperature=args.kd_temp,
            epochs=cfg.student.epochs, lr=cfg.student.lr,
            momentum=cfg.student.momentum, weight_decay=cfg.student.weight_decay,
            output_dir=output_dir, device=args.device,
            start_epoch=start_epoch, initial_best_acc=initial_best_acc,
        )
        save_results(results, os.path.join(output_dir, "results.json"))
        return

    if args.method == "dkd":
        results = train_student_dkd(
            student=student, teacher=teacher_final,
            train_loader=train_loader, test_loader=test_loader,
            epochs=cfg.student.epochs, lr=cfg.student.lr,
            momentum=cfg.student.momentum, weight_decay=cfg.student.weight_decay,
            output_dir=output_dir, device=args.device,
            start_epoch=start_epoch, initial_best_acc=initial_best_acc,
        )
        save_results(results, os.path.join(output_dir, "results.json"))
        return

    if args.method == "crd":
        results = train_student_crd(
            student=student, teacher=teacher_final,
            train_loader=train_loader, test_loader=test_loader,
            student_dim=s_feat_dim, teacher_dim=t_feat_dim,
            alpha_kd=args.alpha_kd, kd_temperature=args.kd_temp,
            epochs=cfg.student.epochs, lr=cfg.student.lr,
            momentum=cfg.student.momentum, weight_decay=cfg.student.weight_decay,
            output_dir=output_dir, device=args.device,
            start_epoch=start_epoch, initial_best_acc=initial_best_acc,
        )
        save_results(results, os.path.join(output_dir, "results.json"))
        return

    # ── ASD methods (need T_early) ──
    teacher_epochs = cfg.teacher.epochs if args.dataset != "imagenet100" else 100
    early_epoch = args.early_epoch or int(teacher_epochs * args.early_ratio)

    early_ckpt = os.path.join(teacher_dir, f"epoch_{early_epoch:04d}.pth")
    if not os.path.exists(early_ckpt):
        # Try nearest checkpoint
        available = sorted([
            int(f.split("_")[1].split(".")[0])
            for f in os.listdir(teacher_dir)
            if f.startswith("epoch_") and f.endswith(".pth") and "best" not in f
        ])
        nearest = min(available, key=lambda x: abs(x - early_epoch))
        early_ckpt = os.path.join(teacher_dir, f"epoch_{nearest:04d}.pth")
        logger.info(f"Exact epoch {early_epoch} not found, using nearest: epoch {nearest}")

    logger.info(f"Loading T_early from: {early_ckpt}")
    teacher_early = load_teacher(args.teacher, early_ckpt, cfg.data.num_classes, args.dataset, args.device)

    logger.info(f"Feature dims - Student: {s_feat_dim}, Teacher: {t_feat_dim}")

    # Configure ASD loss based on method variant
    alpha_tc = args.alpha_tc
    alpha_ss = args.alpha_ss

    if args.method == "asd_tc_only":
        alpha_ss = 0.0
        logger.info("Ablation: TC loss only (SS disabled)")
    elif args.method == "asd_ss_only":
        alpha_tc = 0.0
        logger.info("Ablation: SS loss only (TC disabled)")

    asd_loss_fn = ASDLoss(
        alpha_kd=args.alpha_kd,
        alpha_tc=alpha_tc,
        alpha_ss=alpha_ss,
        kd_temperature=args.kd_temp,
        tc_temperature=args.tc_temp,
        ss_margin=args.ss_margin,
        ss_k_dims=args.ss_k_dims,
        tc_memory_bank_size=args.tc_memory_bank,
        warmup_epochs=args.warmup_epochs,
        student_feat_dim=s_feat_dim,
        teacher_feat_dim=t_feat_dim,
    )

    # Restore projector weights when resuming ASD
    if resume_ckpt_path is not None:
        import torch as _torch
        raw = _torch.load(resume_ckpt_path, map_location=args.device)
        if "loss_state_dict" in raw:
            asd_loss_fn.load_state_dict(raw["loss_state_dict"])
            logger.info("Loaded ASD projector weights from checkpoint")

    results = train_student_asd(
        student=student,
        teacher_final=teacher_final,
        teacher_early=teacher_early,
        train_loader=train_loader,
        test_loader=test_loader,
        asd_loss_fn=asd_loss_fn,
        epochs=cfg.student.epochs,
        lr=cfg.student.lr,
        momentum=cfg.student.momentum,
        weight_decay=cfg.student.weight_decay,
        output_dir=output_dir,
        device=args.device,
        start_epoch=start_epoch,
        initial_best_acc=initial_best_acc,
    )

    save_results(results, os.path.join(output_dir, "results.json"))
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

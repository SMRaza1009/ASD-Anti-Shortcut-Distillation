"""
Training loops for teacher pre-training and ASD student distillation.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict

from core.losses import ASDLoss, BaselineKDLoss, FitNetsLoss, ATLoss, DKDLoss, CRDLoss
from utils.helpers import (
    AverageMeter, MetricTracker, accuracy,
    save_checkpoint, Timer, setup_logging,
)


def get_optimizer(model: nn.Module, lr: float, momentum: float, weight_decay: float):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_scheduler(optimizer, epochs: int, scheduler_type: str = "cosine", start_epoch: int = 0):
    if scheduler_type == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        sched = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    # Fast-forward LR to the correct position for resume
    for _ in range(start_epoch):
        sched.step()
    return sched


# ═══════════════════════════════════════════════════════════
#  Teacher Training
# ═══════════════════════════════════════════════════════════

def train_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 240,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    save_every: int = 10,
    output_dir: str = "outputs/teacher",
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Train teacher model with checkpoint saving for ASD.

    Saves checkpoints every `save_every` epochs for T_early selection.
    """
    logger = setup_logging(output_dir, "teacher_train")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr, momentum, weight_decay)
    scheduler = get_scheduler(optimizer, epochs)
    timer = Timer()

    best_acc = 0
    logger.info(f"Training teacher for {epochs} epochs. Saving checkpoints every {save_every} epochs.")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        tracker = MetricTracker()

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            logits, _ = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, = accuracy(logits, targets, topk=(1,))
            tracker.update({"loss": loss.item(), "acc1": acc1}, n=images.size(0))

        scheduler.step()

        # ── Evaluate ──
        test_acc = evaluate_model(model, test_loader, device)

        # ── Save checkpoint ──
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if epoch % save_every == 0 or is_best or epoch == epochs:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "test_acc": test_acc,
                },
                os.path.join(output_dir, f"epoch_{epoch:04d}.pth"),
                is_best=is_best,
            )

        if epoch % 10 == 0 or epoch == 1:
            s = tracker.summary()
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss {s['loss']:.3f} Acc {s['acc1']:.1f}% | "
                f"Test Acc {test_acc:.1f}% | Best {best_acc:.1f}% | "
                f"LR {scheduler.get_last_lr()[0]:.5f} | {timer.elapsed()}"
            )

    logger.info(f"Teacher training complete. Best accuracy: {best_acc:.2f}%")
    return {"best_acc": best_acc}


# ═══════════════════════════════════════════════════════════
#  Student Training (ASD + baselines)
# ═══════════════════════════════════════════════════════════

def train_student_asd(
    student: nn.Module,
    teacher_final: nn.Module,
    teacher_early: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    asd_loss_fn: ASDLoss,
    epochs: int = 240,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    output_dir: str = "outputs/student",
    device: str = "cuda",
    start_epoch: int = 0,
    initial_best_acc: float = 0.0,
) -> Dict[str, float]:
    """
    Train student model with Anti-Shortcut Distillation.

    Args:
        student: Student model to train
        teacher_final: Frozen fully-converged teacher
        teacher_early: Frozen early-stage teacher checkpoint
        asd_loss_fn: ASD loss module (includes projector if needed)
        train_loader: Training data
        test_loader: Test data
        start_epoch: Resume from this epoch (0 = fresh start)
        initial_best_acc: Best accuracy seen so far (for resume)
    """
    logger = setup_logging(output_dir, "student_train")

    student = student.to(device)
    teacher_final = teacher_final.to(device).eval()
    teacher_early = teacher_early.to(device).eval()
    asd_loss_fn = asd_loss_fn.to(device)

    # Optimizer covers student params + projector params (if any)
    all_params = list(student.parameters()) + list(asd_loss_fn.parameters())
    optimizer = optim.SGD(all_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, epochs, start_epoch=start_epoch)
    timer = Timer()

    best_acc = initial_best_acc
    history = []
    if start_epoch > 0:
        logger.info(f"Resuming ASD training from epoch {start_epoch + 1}/{epochs} (best so far: {best_acc:.2f}%)")
    else:
        logger.info(f"Training student with ASD for {epochs} epochs.")

    for epoch in range(start_epoch + 1, epochs + 1):
        # ── Train ──
        student.train()
        asd_loss_fn.train()
        tracker = MetricTracker()

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            # Student forward
            z_student, h_student = student(images)

            # Teacher forward (no grad)
            with torch.no_grad():
                z_final, h_final = teacher_final(images)
                _, h_early = teacher_early(images)

            # Compute ASD loss
            losses = asd_loss_fn(
                z_student=z_student,
                h_student=h_student,
                z_final=z_final,
                h_final=h_final,
                h_early=h_early,
                targets=targets,
                epoch=epoch,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            acc1, = accuracy(z_student, targets, topk=(1,))
            tracker.update({
                "loss": losses["total"].item(),
                "ce": losses["ce"].item(),
                "kd": losses["kd"].item(),
                "tc": losses["tc"].item(),
                "ss": losses["ss"].item(),
                "acc1": acc1,
            }, n=images.size(0))

        scheduler.step()

        # ── Evaluate ──
        test_acc = evaluate_model(student, test_loader, device)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        s = tracker.summary()
        epoch_record = {"epoch": epoch, "test_acc": test_acc, **s}
        history.append(epoch_record)

        if epoch % save_every_n(epochs) == 0 or is_best or epoch == epochs:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": student.state_dict(),
                    "loss_state_dict": asd_loss_fn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "test_acc": test_acc,
                },
                os.path.join(output_dir, f"epoch_{epoch:04d}.pth"),
                is_best=is_best,
            )

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss {s['loss']:.3f} (CE:{s['ce']:.3f} KD:{s['kd']:.3f} "
                f"TC:{s['tc']:.3f} SS:{s['ss']:.3f}) | "
                f"Train {s['acc1']:.1f}% | Test {test_acc:.1f}% | Best {best_acc:.1f}% | "
                f"{timer.elapsed()}"
            )

    logger.info(f"Student training complete. Best accuracy: {best_acc:.2f}%")
    return {"best_acc": best_acc, "history": history}


def train_student_vanilla_kd(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    alpha_kd: float = 1.0,
    kd_temperature: float = 4.0,
    epochs: int = 240,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    output_dir: str = "outputs/student_kd",
    device: str = "cuda",
    start_epoch: int = 0,
    initial_best_acc: float = 0.0,
) -> Dict[str, float]:
    """Standard vanilla KD training (baseline)."""
    logger = setup_logging(output_dir, "student_kd")
    student = student.to(device)
    teacher = teacher.to(device).eval()

    loss_fn = BaselineKDLoss(alpha_kd, kd_temperature)
    optimizer = get_optimizer(student, lr, momentum, weight_decay)
    scheduler = get_scheduler(optimizer, epochs, start_epoch=start_epoch)
    timer = Timer()

    best_acc = initial_best_acc
    if start_epoch > 0:
        logger.info(f"Resuming vanilla KD from epoch {start_epoch + 1}/{epochs} (best: {best_acc:.2f}%)")
    else:
        logger.info(f"Training student with vanilla KD for {epochs} epochs.")

    for epoch in range(start_epoch + 1, epochs + 1):
        student.train()
        tracker = MetricTracker()

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            z_student, _ = student(images)

            with torch.no_grad():
                z_teacher, _ = teacher(images)

            losses = loss_fn(z_student, z_teacher, targets)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            acc1, = accuracy(z_student, targets, topk=(1,))
            tracker.update({"loss": losses["total"].item(), "acc1": acc1}, n=images.size(0))

        scheduler.step()
        test_acc = evaluate_model(student, test_loader, device)
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if epoch % save_every_n(epochs) == 0 or is_best or epoch == epochs:
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": student.state_dict(), "best_acc": best_acc},
                os.path.join(output_dir, f"epoch_{epoch:04d}.pth"),
                is_best=is_best,
            )

        if epoch % 10 == 0 or epoch == 1:
            s = tracker.summary()
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | Loss {s['loss']:.3f} | "
                f"Train {s['acc1']:.1f}% | Test {test_acc:.1f}% | Best {best_acc:.1f}% | {timer.elapsed()}"
            )

    logger.info(f"Vanilla KD training complete. Best: {best_acc:.2f}%")
    return {"best_acc": best_acc}


def train_student_ce_only(
    student: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 240,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    output_dir: str = "outputs/student_ce",
    device: str = "cuda",
    start_epoch: int = 0,
    initial_best_acc: float = 0.0,
) -> Dict[str, float]:
    """Train student with cross-entropy only (no distillation)."""
    logger = setup_logging(output_dir, "student_ce")
    student = student.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(student, lr, momentum, weight_decay)
    scheduler = get_scheduler(optimizer, epochs, start_epoch=start_epoch)

    best_acc = initial_best_acc
    if start_epoch > 0:
        logger.info(f"Resuming CE-only from epoch {start_epoch + 1}/{epochs} (best: {best_acc:.2f}%)")
    for epoch in range(start_epoch + 1, epochs + 1):
        student.train()
        tracker = MetricTracker()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            logits, _ = student(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1, = accuracy(logits, targets, topk=(1,))
            tracker.update({"loss": loss.item(), "acc1": acc1}, n=images.size(0))

        scheduler.step()
        test_acc = evaluate_model(student, test_loader, device)
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if is_best or epoch == epochs:
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": student.state_dict(), "best_acc": best_acc},
                os.path.join(output_dir, f"epoch_{epoch:04d}.pth"),
                is_best=is_best,
            )

        if epoch % 10 == 0:
            s = tracker.summary()
            logger.info(f"Epoch {epoch:3d}/{epochs} | Test {test_acc:.1f}% | Best {best_acc:.1f}%")

    return {"best_acc": best_acc}


# ═══════════════════════════════════════════════════════════
#  Baseline trainers
# ═══════════════════════════════════════════════════════════

def train_student_fitnets(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    student_dim: int = 512,
    teacher_dim: int = 512,
    alpha_kd: float = 1.0,
    kd_temperature: float = 4.0,
    epochs: int = 240,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    output_dir: str = "outputs/student_fitnets",
    device: str = "cuda",
    start_epoch: int = 0,
    initial_best_acc: float = 0.0,
) -> Dict[str, float]:
    """FitNets: hint-layer feature matching + logit KD."""
    logger = setup_logging(output_dir, "student_fitnets")
    student = student.to(device)
    teacher = teacher.to(device).eval()

    fitnets_loss = FitNetsLoss(student_dim, teacher_dim).to(device)
    kd_loss = BaselineKDLoss(alpha_kd, kd_temperature)

    all_params = list(student.parameters()) + list(fitnets_loss.parameters())
    optimizer = optim.SGD(all_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, epochs, start_epoch=start_epoch)
    timer = Timer()
    best_acc = initial_best_acc

    if start_epoch > 0:
        logger.info(f"Resuming FitNets from epoch {start_epoch + 1}/{epochs} (best: {best_acc:.2f}%)")
    else:
        logger.info(f"Training student with FitNets for {epochs} epochs.")
    for epoch in range(start_epoch + 1, epochs + 1):
        student.train()
        fitnets_loss.train()
        tracker = MetricTracker()

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            z_s, h_s = student(images)
            with torch.no_grad():
                z_t, h_t = teacher(images)

            losses_kd = kd_loss(z_s, z_t, targets)
            loss_hint = fitnets_loss(h_s, h_t)
            loss = losses_kd["total"] + loss_hint

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1, = accuracy(z_s, targets, topk=(1,))
            tracker.update({"loss": loss.item(), "acc1": acc1}, n=images.size(0))

        scheduler.step()
        test_acc = evaluate_model(student, test_loader, device)
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if epoch % save_every_n(epochs) == 0 or is_best or epoch == epochs:
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": student.state_dict(), "best_acc": best_acc},
                os.path.join(output_dir, f"epoch_{epoch:04d}.pth"), is_best=is_best,
            )
        if epoch % 10 == 0 or epoch == 1:
            s = tracker.summary()
            logger.info(f"Epoch {epoch:3d}/{epochs} | Loss {s['loss']:.3f} | "
                        f"Train {s['acc1']:.1f}% | Test {test_acc:.1f}% | Best {best_acc:.1f}% | {timer.elapsed()}")

    logger.info(f"FitNets training complete. Best: {best_acc:.2f}%")
    return {"best_acc": best_acc}


def train_student_at(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    alpha_kd: float = 1.0,
    kd_temperature: float = 4.0,
    at_beta: float = 1000.0,
    epochs: int = 240,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    output_dir: str = "outputs/student_at",
    device: str = "cuda",
    start_epoch: int = 0,
    initial_best_acc: float = 0.0,
) -> Dict[str, float]:
    """Attention Transfer: matches spatial attention maps between student and teacher."""
    logger = setup_logging(output_dir, "student_at")
    student = student.to(device)
    teacher = teacher.to(device).eval()
    at_loss_fn = ATLoss(beta=at_beta)
    kd_loss = BaselineKDLoss(alpha_kd, kd_temperature)

    optimizer = get_optimizer(student, lr, momentum, weight_decay)
    scheduler = get_scheduler(optimizer, epochs, start_epoch=start_epoch)
    timer = Timer()
    best_acc = initial_best_acc

    if start_epoch > 0:
        logger.info(f"Resuming AT from epoch {start_epoch + 1}/{epochs} (best: {best_acc:.2f}%)")
    else:
        logger.info(f"Training student with Attention Transfer for {epochs} epochs.")
    for epoch in range(start_epoch + 1, epochs + 1):
        student.train()
        tracker = MetricTracker()

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            z_s, h_s = student(images)
            s_all = student.extract_features_all_layers(images)
            with torch.no_grad():
                z_t, h_t = teacher(images)
                t_all = teacher.extract_features_all_layers(images)

            losses_kd = kd_loss(z_s, z_t, targets)
            loss_at = at_loss_fn(s_all, t_all)
            loss = losses_kd["total"] + loss_at

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1, = accuracy(z_s, targets, topk=(1,))
            tracker.update({"loss": loss.item(), "acc1": acc1}, n=images.size(0))

        scheduler.step()
        test_acc = evaluate_model(student, test_loader, device)
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if epoch % save_every_n(epochs) == 0 or is_best or epoch == epochs:
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": student.state_dict(), "best_acc": best_acc},
                os.path.join(output_dir, f"epoch_{epoch:04d}.pth"), is_best=is_best,
            )
        if epoch % 10 == 0 or epoch == 1:
            s = tracker.summary()
            logger.info(f"Epoch {epoch:3d}/{epochs} | Loss {s['loss']:.3f} | "
                        f"Train {s['acc1']:.1f}% | Test {test_acc:.1f}% | Best {best_acc:.1f}% | {timer.elapsed()}")

    logger.info(f"AT training complete. Best: {best_acc:.2f}%")
    return {"best_acc": best_acc}


def train_student_dkd(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float = 4.0,
    warmup_epochs: int = 20,
    epochs: int = 240,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    output_dir: str = "outputs/student_dkd",
    device: str = "cuda",
    start_epoch: int = 0,
    initial_best_acc: float = 0.0,
) -> Dict[str, float]:
    """Decoupled Knowledge Distillation (Zhao et al., NeurIPS 2022)."""
    logger = setup_logging(output_dir, "student_dkd")
    student = student.to(device)
    teacher = teacher.to(device).eval()
    dkd_loss_fn = DKDLoss(alpha, beta, temperature, warmup_epochs)

    optimizer = get_optimizer(student, lr, momentum, weight_decay)
    scheduler = get_scheduler(optimizer, epochs, start_epoch=start_epoch)
    timer = Timer()
    best_acc = initial_best_acc

    if start_epoch > 0:
        logger.info(f"Resuming DKD from epoch {start_epoch + 1}/{epochs} (best: {best_acc:.2f}%)")
    else:
        logger.info(f"Training student with DKD for {epochs} epochs.")
    for epoch in range(start_epoch + 1, epochs + 1):
        student.train()
        tracker = MetricTracker()

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            z_s, _ = student(images)
            with torch.no_grad():
                z_t, _ = teacher(images)

            losses = dkd_loss_fn(z_s, z_t, targets, epoch)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            acc1, = accuracy(z_s, targets, topk=(1,))
            tracker.update({"loss": losses["total"].item(), "acc1": acc1}, n=images.size(0))

        scheduler.step()
        test_acc = evaluate_model(student, test_loader, device)
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if epoch % save_every_n(epochs) == 0 or is_best or epoch == epochs:
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": student.state_dict(), "best_acc": best_acc},
                os.path.join(output_dir, f"epoch_{epoch:04d}.pth"), is_best=is_best,
            )
        if epoch % 10 == 0 or epoch == 1:
            s = tracker.summary()
            logger.info(f"Epoch {epoch:3d}/{epochs} | Loss {s['loss']:.3f} | "
                        f"Train {s['acc1']:.1f}% | Test {test_acc:.1f}% | Best {best_acc:.1f}% | {timer.elapsed()}")

    logger.info(f"DKD training complete. Best: {best_acc:.2f}%")
    return {"best_acc": best_acc}


def train_student_crd(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    student_dim: int = 512,
    teacher_dim: int = 512,
    alpha_kd: float = 1.0,
    kd_temperature: float = 4.0,
    crd_temperature: float = 0.07,
    crd_weight: float = 0.8,
    epochs: int = 240,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    output_dir: str = "outputs/student_crd",
    device: str = "cuda",
    start_epoch: int = 0,
    initial_best_acc: float = 0.0,
) -> Dict[str, float]:
    """
    Contrastive Representation Distillation — in-batch variant.
    Negatives: other samples in the mini-batch (different from ASD's time-state negatives).
    """
    logger = setup_logging(output_dir, "student_crd")
    student = student.to(device)
    teacher = teacher.to(device).eval()

    crd_loss_fn = CRDLoss(crd_temperature, student_dim, teacher_dim).to(device)
    kd_loss = BaselineKDLoss(alpha_kd, kd_temperature)

    all_params = list(student.parameters()) + list(crd_loss_fn.parameters())
    optimizer = optim.SGD(all_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, epochs, start_epoch=start_epoch)
    timer = Timer()
    best_acc = initial_best_acc

    if start_epoch > 0:
        logger.info(f"Resuming CRD from epoch {start_epoch + 1}/{epochs} (best: {best_acc:.2f}%)")
    else:
        logger.info(f"Training student with CRD (in-batch) for {epochs} epochs.")
    for epoch in range(start_epoch + 1, epochs + 1):
        student.train()
        crd_loss_fn.train()
        tracker = MetricTracker()

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            z_s, h_s = student(images)
            with torch.no_grad():
                z_t, h_t = teacher(images)

            losses_kd = kd_loss(z_s, z_t, targets)
            loss_crd = crd_loss_fn(h_s, h_t)
            loss = losses_kd["total"] + crd_weight * loss_crd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1, = accuracy(z_s, targets, topk=(1,))
            tracker.update({"loss": loss.item(), "acc1": acc1}, n=images.size(0))

        scheduler.step()
        test_acc = evaluate_model(student, test_loader, device)
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if epoch % save_every_n(epochs) == 0 or is_best or epoch == epochs:
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": student.state_dict(), "best_acc": best_acc},
                os.path.join(output_dir, f"epoch_{epoch:04d}.pth"), is_best=is_best,
            )
        if epoch % 10 == 0 or epoch == 1:
            s = tracker.summary()
            logger.info(f"Epoch {epoch:3d}/{epochs} | Loss {s['loss']:.3f} | "
                        f"Train {s['acc1']:.1f}% | Test {test_acc:.1f}% | Best {best_acc:.1f}% | {timer.elapsed()}")

    logger.info(f"CRD training complete. Best: {best_acc:.2f}%")
    return {"best_acc": best_acc}


# ═══════════════════════════════════════════════════════════
#  Evaluation helpers
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    """Evaluate model accuracy on a dataloader."""
    model.eval()
    correct, total = 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits, _ = model(images)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def save_every_n(total_epochs: int) -> int:
    """Determine checkpoint save frequency."""
    if total_epochs <= 100:
        return 25
    return 50

"""
Loss functions for ASD and all baseline KD methods.

Novel (ASD):
  L_TC  — Temporal Contrastive Loss
  L_SS  — Shortcut Suppression Loss

Baselines:
  FitNetsLoss  — Hint-based feature matching (Romero et al. 2015)
  ATLoss       — Attention Transfer (Zagoruyko & Komodakis, ICLR 2017)
  DKDLoss      — Decoupled KD (Zhao et al., NeurIPS 2022)
  CRDLoss      — Contrastive Representation Distillation, in-batch variant (Tian et al., ICLR 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TemporalContrastiveLoss(nn.Module):
    """
    InfoNCE loss combining temporal negatives (h_early) with in-batch negatives
    and an optional cross-batch FIFO memory bank (Fix B).

    Positive: h_final of same sample (robust final teacher features)
    Negatives: h_early of same sample (temporal shortcut negative)
               + h_final of other batch samples (in-batch diversity, like CRD)
               + memory bank: h_final from previous batches (cross-batch diversity)

    memory_bank_size=0 disables the bank (in-batch only, original behavior).
    memory_bank_size=4096 adds ~64 extra batches of negatives, matching CRD scale.
    """

    def __init__(self, temperature: float = 0.07, memory_bank_size: int = 0, feat_dim: int = 512):
        super().__init__()
        self.temperature = temperature
        self.memory_bank_size = memory_bank_size
        if memory_bank_size > 0:
            # FIFO queue: stores normalized h_final features from previous batches.
            # Registered as buffer so .to(device) and state_dict work automatically.
            queue = F.normalize(torch.randn(memory_bank_size, feat_dim), dim=-1)
            self.register_buffer("memory_queue", queue)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, h_final_norm: torch.Tensor) -> None:
        B = h_final_norm.size(0)
        ptr = int(self.queue_ptr)
        # Wrap-around FIFO write
        space = self.memory_bank_size - ptr
        if B <= space:
            self.memory_queue[ptr:ptr + B] = h_final_norm
        else:
            self.memory_queue[ptr:] = h_final_norm[:space]
            self.memory_queue[:B - space] = h_final_norm[space:]
        self.queue_ptr[0] = (ptr + B) % self.memory_bank_size

    def forward(
        self,
        h_student: torch.Tensor,    # (B, D_s)
        h_final: torch.Tensor,      # (B, D_t)
        h_early: torch.Tensor,      # (B, D_t)
        projector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if projector is not None:
            h_student = projector(h_student)

        B = h_student.size(0)
        h_s = F.normalize(h_student, dim=-1)   # (B, D)
        h_f = F.normalize(h_final, dim=-1)     # (B, D)
        h_e = F.normalize(h_early, dim=-1)     # (B, D)

        # Positive: same-sample final teacher similarity
        pos_sim = (h_s * h_f).sum(dim=-1, keepdim=True) / self.temperature  # (B, 1)

        # Temporal negative: same-sample early teacher similarity
        temporal_neg = (h_s * h_e).sum(dim=-1, keepdim=True) / self.temperature  # (B, 1)

        # In-batch negatives: h_final of all other samples (B-1 negatives)
        inbatch_sim = torch.mm(h_s, h_f.T) / self.temperature  # (B, B)
        mask = torch.eye(B, dtype=torch.bool, device=h_s.device)
        inbatch_neg = inbatch_sim[~mask].view(B, B - 1)  # (B, B-1)

        if self.memory_bank_size > 0:
            # Clone before enqueue: _enqueue modifies memory_queue in-place, which would
            # increment the storage version and break autograd's saved-tensor check.
            queue = self.memory_queue.detach().clone()  # own storage, no grad_fn
            mem_neg = torch.mm(h_s, queue.T) / self.temperature  # (B, K)
            logits = torch.cat([pos_sim, temporal_neg, inbatch_neg, mem_neg], dim=1)
            self._enqueue(h_f.detach())
        else:
            logits = torch.cat([pos_sim, temporal_neg, inbatch_neg], dim=1)  # (B, B+1)

        labels = torch.zeros(B, dtype=torch.long, device=h_s.device)
        return F.cross_entropy(logits, labels)


class ShortcutSuppressionLoss(nn.Module):
    """
    Penalizes student for activating along the shortcut subspace.

    Shortcut subspace = top-k PCA eigenvectors of Cov(Δh), where
    Δh = h_early − h_final captures what the teacher learned to suppress.

    The per-sample covariance captures the principal axes of the shortcut
    feature space rather than a single batch-mean direction that collapses
    to near-zero in high-dim space (previous batch-mean approach gave SS≈0.003).

    L_SS = mean( ReLU( ||Proj(h_s, V_k)||_2 − ε ) )

    where V_k ∈ R^{D×k} are the top-k eigenvectors of Cov(Δh).
    For random unit vectors in D-dim: E[||proj||_2] ≈ sqrt(k/D), so
    margin ε slightly above this level fires only for non-random alignment.
    """

    def __init__(self, margin: float = 0.1, k_dims: int = 4):
        super().__init__()
        self.margin = margin
        self.k_dims = k_dims

    def forward(
        self,
        h_student: torch.Tensor,    # (B, D_s)
        h_final: torch.Tensor,      # (B, D_t)
        h_early: torch.Tensor,      # (B, D_t)
        projector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if projector is not None:
            h_student = projector(h_student)

        # Per-sample shortcut delta; detach since teacher is frozen
        delta = (h_early - h_final).detach()          # (B, D)
        # Do NOT center: the mean direction of delta IS the shortcut direction.
        # Centering would subtract it away.

        # Top-k right singular vectors of delta == top-k eigenvectors of delta.T @ delta.
        # Using SVD on (B, D) is O(B²D) vs O(D³) for eigh on the full (D, D) covariance,
        # and avoids allocating the large covariance matrix entirely.
        k = min(self.k_dims, delta.size(0))
        try:
            _, _, Vh = torch.linalg.svd(delta, full_matrices=False)  # Vh: (min(B,D), D)
            shortcut_dirs = Vh[:k].T                  # (D, k) top-k right singular vectors
        except RuntimeError:
            # Fallback to single normalized mean direction if SVD fails
            shortcut_dirs = F.normalize(
                delta.mean(0, keepdim=True), dim=-1
            ).T                                       # (D, 1)

        # Student projection magnitude onto shortcut subspace
        h_s = F.normalize(h_student, dim=-1)          # (B, D)
        proj = h_s @ shortcut_dirs                    # (B, k)
        proj_norm = proj.pow(2).sum(dim=-1).sqrt()    # (B,)  L2 subspace projection

        return F.relu(proj_norm - self.margin).mean()


class VanillaKDLoss(nn.Module):
    """Standard knowledge distillation loss (Hinton et al., 2015)."""

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_student: torch.Tensor,  # Student logits (B, C)
        z_teacher: torch.Tensor,  # Teacher logits (B, C)
    ) -> torch.Tensor:
        p_s = F.log_softmax(z_student / self.temperature, dim=-1)
        p_t = F.softmax(z_teacher / self.temperature, dim=-1)
        loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.temperature ** 2)
        return loss


class ASDLoss(nn.Module):
    """
    Complete Anti-Shortcut Distillation loss.

    L_total = L_CE + α·L_KD + β·L_TC + γ·L_SS

    Where β and γ are linearly warmed up over the first few epochs.
    """

    def __init__(
        self,
        alpha_kd: float = 1.0,
        alpha_tc: float = 0.8,
        alpha_ss: float = 1.0,
        kd_temperature: float = 4.0,
        tc_temperature: float = 0.07,
        ss_margin: float = 0.1,
        ss_k_dims: int = 4,
        tc_memory_bank_size: int = 4096,
        warmup_epochs: int = 20,
        student_feat_dim: int = 512,
        teacher_feat_dim: int = 512,
    ):
        super().__init__()
        self.alpha_kd = alpha_kd
        self.alpha_tc = alpha_tc
        self.alpha_ss = alpha_ss
        self.warmup_epochs = warmup_epochs

        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = VanillaKDLoss(kd_temperature)
        self.tc_loss = TemporalContrastiveLoss(tc_temperature, tc_memory_bank_size, teacher_feat_dim)
        self.ss_loss = ShortcutSuppressionLoss(ss_margin, ss_k_dims)

        # Feature projector: map student features to teacher feature space
        # Architecture per report Section 4.1: Linear → BN → ReLU → Linear
        if student_feat_dim != teacher_feat_dim:
            self.projector = nn.Sequential(
                nn.Linear(student_feat_dim, teacher_feat_dim),
                nn.BatchNorm1d(teacher_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(teacher_feat_dim, teacher_feat_dim),
            )
        else:
            self.projector = None

    def _warmup_factor(self, epoch: int) -> float:
        """Linear warmup from 0 to 1 over warmup_epochs."""
        if epoch >= self.warmup_epochs:
            return 1.0
        return epoch / self.warmup_epochs

    def forward(
        self,
        z_student: torch.Tensor,   # Student logits
        h_student: torch.Tensor,   # Student features
        z_final: torch.Tensor,     # Final teacher logits
        h_final: torch.Tensor,     # Final teacher features
        h_early: torch.Tensor,     # Early teacher features
        targets: torch.Tensor,     # Ground truth labels
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components and return dict.

        Returns dict with keys: 'total', 'ce', 'kd', 'tc', 'ss'
        """
        warmup = self._warmup_factor(epoch)

        # L_CE: standard cross-entropy
        loss_ce = self.ce_loss(z_student, targets)

        # L_KD: standard KD (logit matching with T_final)
        loss_kd = self.kd_loss(z_student, z_final)

        # L_TC: temporal contrastive loss
        loss_tc = self.tc_loss(h_student, h_final, h_early, self.projector)

        # L_SS: shortcut suppression loss
        loss_ss = self.ss_loss(h_student, h_final, h_early, self.projector)

        # Total with warmup on novel losses
        loss_total = (
            loss_ce
            + self.alpha_kd * loss_kd
            + warmup * self.alpha_tc * loss_tc
            + warmup * self.alpha_ss * loss_ss
        )

        return {
            "total": loss_total,
            "ce": loss_ce.detach(),
            "kd": loss_kd.detach(),
            "tc": loss_tc.detach(),
            "ss": loss_ss.detach(),
        }


# ═══════════════════════════════════════════════════════════
#  Baseline KD Methods
# ═══════════════════════════════════════════════════════════

class FitNetsLoss(nn.Module):
    """
    Hint-based feature matching (Romero et al., 2015).
    L_FitNets = (1/2n) * ||W_r(h_student) - h_teacher||^2_F
    A linear regressor W_r maps student hint layer to teacher guided layer.
    """
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.ReLU(inplace=True),
            nn.Linear(teacher_dim, teacher_dim),
        )

    def forward(
        self,
        h_student: torch.Tensor,   # (B, D_s)
        h_teacher: torch.Tensor,   # (B, D_t)
    ) -> torch.Tensor:
        h_projected = self.regressor(h_student)
        return F.mse_loss(h_projected, h_teacher.detach())


class ATLoss(nn.Module):
    """
    Attention Transfer (Zagoruyko & Komodakis, ICLR 2017).
    Matches spatial attention maps between student and teacher feature maps.
    L_AT = sum_i beta * ||Q(F_s_i) - Q(F_t_i)||_2^2
    where Q(F) = L2-normalize(sum_c |F_c|^p).
    """
    def __init__(self, p: float = 2.0, beta: float = 1000.0):
        super().__init__()
        self.p = p
        self.beta = beta

    @staticmethod
    def attention_map(feat: torch.Tensor, p: float) -> torch.Tensor:
        """Spatial attention map: (B, C, H, W) -> (B, H*W) normalized."""
        attn = feat.pow(p).mean(dim=1)          # (B, H, W)
        attn = attn.view(attn.size(0), -1)       # (B, H*W)
        return F.normalize(attn, dim=1)

    def forward(
        self,
        student_feats: Dict[str, torch.Tensor],  # layer_name -> (B, C, H, W)
        teacher_feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(iter(student_feats.values())).device)
        shared_layers = set(student_feats.keys()) & set(teacher_feats.keys())
        shared_layers.discard("penultimate")
        for layer in shared_layers:
            fs = student_feats[layer]
            ft = teacher_feats[layer]
            # Resize student map to teacher spatial size if needed
            if fs.shape[2:] != ft.shape[2:]:
                fs = F.adaptive_avg_pool2d(fs, ft.shape[2:])
            As = self.attention_map(fs, self.p)
            At = self.attention_map(ft, self.p)
            loss = loss + (As - At).pow(2).mean()
        return self.beta * loss


class DKDLoss(nn.Module):
    """
    Decoupled Knowledge Distillation (Zhao et al., NeurIPS 2022).
    Decomposes KD into:
      TCKD  — Target Class KD: distillation on the target-class logit
      NCKD  — Non-Target Class KD: distillation on non-target logits (conditioned)
    L_DKD = alpha * L_TCKD + beta * L_NCKD
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 4.0,
        warmup_epochs: int = 20,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.warmup_epochs = warmup_epochs

    def _tckd(self, z_s, z_t, targets, T):
        """Target Class KD."""
        p_s = F.softmax(z_s / T, dim=1)
        p_t = F.softmax(z_t / T, dim=1)
        # Binary cross entropy on target-class probabilities
        p_s_tgt = p_s.gather(1, targets.unsqueeze(1))
        p_t_tgt = p_t.gather(1, targets.unsqueeze(1))
        loss = -(p_t_tgt * torch.log(p_s_tgt + 1e-8) +
                 (1 - p_t_tgt) * torch.log(1 - p_s_tgt + 1e-8)).mean()
        return (T ** 2) * loss

    def _nckd(self, z_s, z_t, targets, T):
        """Non-Target Class KD — KL on non-target logits."""
        B, C = z_s.shape
        # Mask out target class
        mask = torch.ones(B, C, dtype=torch.bool, device=z_s.device)
        mask.scatter_(1, targets.unsqueeze(1), False)
        z_s_nt = z_s[mask].view(B, C - 1)
        z_t_nt = z_t[mask].view(B, C - 1)
        p_s = F.log_softmax(z_s_nt / T, dim=1)
        p_t = F.softmax(z_t_nt / T, dim=1)
        return (T ** 2) * F.kl_div(p_s, p_t, reduction="batchmean")

    def forward(
        self,
        z_student: torch.Tensor,
        z_teacher: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        T = self.temperature
        warmup = min(1.0, epoch / max(self.warmup_epochs, 1))
        loss_ce = F.cross_entropy(z_student, targets)
        loss_tckd = self._tckd(z_student, z_teacher, targets, T)
        loss_nckd = self._nckd(z_student, z_teacher, targets, T)
        loss_total = loss_ce + self.alpha * loss_tckd + warmup * self.beta * loss_nckd
        return {
            "total": loss_total,
            "ce": loss_ce.detach(),
            "tckd": loss_tckd.detach(),
            "nckd": loss_nckd.detach(),
        }


class CRDLoss(nn.Module):
    """
    Contrastive Representation Distillation — in-batch variant.
    (Tian et al., ICLR 2020)

    Key distinction from ASD: negatives are *different samples* in the batch,
    not different time-states of the teacher. This is used as a baseline to show
    ASD's temporal negatives outperform sample-level negatives.

    L_CRD = InfoNCE(h_s_i, h_t_i as positive, h_t_j (j!=i) as negatives)
    """
    def __init__(self, temperature: float = 0.07, student_dim: int = 512, teacher_dim: int = 512):
        super().__init__()
        self.temperature = temperature
        if student_dim != teacher_dim:
            self.projector = nn.Sequential(
                nn.Linear(student_dim, teacher_dim),
                nn.ReLU(inplace=True),
                nn.Linear(teacher_dim, teacher_dim),
            )
        else:
            self.projector = None

    def forward(
        self,
        h_student: torch.Tensor,   # (B, D_s)
        h_teacher: torch.Tensor,   # (B, D_t)
    ) -> torch.Tensor:
        if self.projector is not None:
            h_student = self.projector(h_student)
        h_s = F.normalize(h_student, dim=1)   # (B, D)
        h_t = F.normalize(h_teacher, dim=1)   # (B, D)
        # Similarity matrix: (B, B) — diagonal is positive pair
        sim = torch.mm(h_s, h_t.T) / self.temperature   # (B, B)
        labels = torch.arange(h_s.size(0), device=h_s.device)
        return F.cross_entropy(sim, labels)


class BaselineKDLoss(nn.Module):
    """Standard KD for baseline comparison: L_CE + α·L_KD."""

    def __init__(self, alpha_kd: float = 1.0, kd_temperature: float = 4.0):
        super().__init__()
        self.alpha_kd = alpha_kd
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = VanillaKDLoss(kd_temperature)

    def forward(
        self,
        z_student: torch.Tensor,
        z_teacher: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        loss_ce = self.ce_loss(z_student, targets)
        loss_kd = self.kd_loss(z_student, z_teacher)
        loss_total = loss_ce + self.alpha_kd * loss_kd
        return {
            "total": loss_total,
            "ce": loss_ce.detach(),
            "kd": loss_kd.detach(),
            "tc": torch.tensor(0.0),
            "ss": torch.tensor(0.0),
        }

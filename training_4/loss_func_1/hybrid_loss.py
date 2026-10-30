"""Normalized per-example AR + prefix-distance CE, using PyTorch operations."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from loss_func_1.config_loss import HybridLossConfig


class HybridLoss(nn.Module):
    def __init__(self, config: HybridLossConfig):
        super().__init__()
        self.config = config

    def half_life(self, completed_steps: int, total_steps: int) -> float:
        if completed_steps < 0 or total_steps <= 0:
            raise ValueError("Require completed_steps >= 0 and total_steps > 0")
        s = min(completed_steps / total_steps, 1.0)
        return self.config.half_life_start + s * (
            self.config.half_life_end - self.config.half_life_start)

    def token_ce(self, logits, labels):
        if logits.shape[:-1] != labels.shape or logits.device != labels.device:
            raise ValueError("Logits and labels must have matching shapes/devices")
        # FP32 reduction/softmax even when forward uses a reduced precision dtype.
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=self.config.ignore_index, reduction="none",
        ).reshape_as(labels)

    def ar_loss(self, logits, labels):
        """labels are UNshifted clean targets; caller MUST use causal logits."""
        targets = labels[:, 1:]
        valid = targets.ne(self.config.ignore_index)
        counts = valid.sum(dim=-1)
        if torch.any(counts == 0):
            raise ValueError("Every example needs at least one AR target")
        ce = self.token_ce(logits[:, :-1], targets)
        return (ce.sum(dim=-1) / counts).mean()

    def distance_weights(self, labels, prefix_end, completed_steps, total_steps):
        """prefix_end is a ZERO-BASED absolute index, not masked-token rank."""
        valid = labels.ne(self.config.ignore_index)
        counts = valid.sum(dim=-1, keepdim=True)
        if prefix_end.shape != (labels.shape[0],) or prefix_end.device != labels.device:
            raise ValueError("prefix_end must have shape [B] on labels.device")
        if torch.any(counts == 0):
            raise ValueError("Empty denoising target set; apply the masking policy first")
        positions = torch.arange(labels.size(1), device=labels.device)
        distances = positions[None, :] - prefix_end[:, None]
        if torch.any(valid & (distances < 1)):
            raise ValueError("Denoising targets must be after the prefix")
        # Subtract nearest masked distance before scaling: softmax-invariant,
        # and avoids all scores overflowing for a very small half-life.
        nearest = distances.masked_fill(~valid, labels.size(1) + 1).amin(-1, keepdim=True)
        offsets = (distances - nearest).clamp_min(0).float()
        scores = -offsets * math.log(2) / self.half_life(completed_steps, total_steps)
        local = torch.softmax(scores.masked_fill(~valid, -torch.inf), dim=-1)
        uniform = valid.float() / counts
        return self.config.rho * uniform + (1 - self.config.rho) * local

    def distance_loss(self, logits, labels, prefix_end, completed_steps, total_steps):
        q = self.distance_weights(labels, prefix_end, completed_steps, total_steps)
        return (q * self.token_ce(logits, labels)).sum(dim=-1).mean()

    def forward(self, ar_logits, ar_labels, den_logits, den_labels,
                prefix_end, completed_steps, total_steps):
        ar = self.ar_loss(ar_logits, ar_labels)
        distance = self.distance_loss(den_logits, den_labels, prefix_end,
                                      completed_steps, total_steps)
        return self.config.alpha * ar + (1 - self.config.alpha) * distance

"""
Loss functions.

Currently: MSE (data loss only) for vanilla DeepONet.
Later: physics residual loss for PI-DeepONet.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Standard MSE loss for data-driven DeepONet."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)


class RelativeL2Loss(nn.Module):
    """
    Relative L2 error — better metric than raw MSE for comparing across scales.
    ||pred - target||_2 / ||target||_2
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.norm(pred - target) / (torch.norm(target) + 1e-8)


# Registry for config-driven loss selection
LOSSES = {
    "mse": MSELoss,
    "relative_l2": RelativeL2Loss,
}


def get_loss(name: str) -> nn.Module:
    if name not in LOSSES:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSSES.keys())}")
    return LOSSES[name]()

"""
DeepONet: combines branch and trunk nets.

Forward pass:
    branch_out = branch(u)          # (batch, p)
    trunk_out  = trunk(y)           # (batch, p)
    output     = sum(branch * trunk, dim=-1) + bias   # (batch,)

The bias term (scalar, learnable) is from the original Lu et al. paper —
it improves training stability.
"""

import torch
import torch.nn as nn
from typing import List

from models.branch import BranchNet
from models.trunk import TrunkNet


class DeepONet(nn.Module):
    """
    Vanilla DeepONet (unstacked, single output).

    Args:
        m: sensor points (branch input size)
        d_y: query point dimension (trunk input size)
        hidden_dims_branch: hidden layer sizes for branch net
        hidden_dims_trunk: hidden layer sizes for trunk net
        p: shared latent dimension
        activation: activation function
    """

    def __init__(
        self,
        m: int,
        d_y: int,
        hidden_dims_branch: List[int],
        hidden_dims_trunk: List[int],
        p: int,
        activation: str = "tanh",
    ):
        super().__init__()

        self.branch = BranchNet(m, hidden_dims_branch, p, activation)
        self.trunk  = TrunkNet(d_y, hidden_dims_trunk, p, activation)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, m) — input function at sensor points
            y: (batch, d_y) — query locations
        Returns:
            s: (batch, 1) — predicted operator output at y
        """
        b = self.branch(u)                          # (batch, p)
        t = self.trunk(y)                           # (batch, p)
        out = (b * t).sum(dim=-1, keepdim=True)    # (batch, 1)
        return out + self.bias

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DeepONet(
        m=100, d_y=1,
        hidden_dims_branch=[128, 128, 128],
        hidden_dims_trunk=[128, 128, 128],
        p=128,
    )
    print(f"Parameters: {model.count_params():,}")

    u = torch.randn(32, 100)
    y = torch.rand(32, 1)
    out = model(u, y)
    print(f"Output shape: {out.shape}")   # (32, 1)

"""
Branch network: encodes the input function u.

Input:  u evaluated at m fixed sensor points → (batch, m)
Output: vector of size p → (batch, p)

This is just an MLP. The key point is that all input functions
must use the same m sensor locations — this is a hard architectural constraint.
"""

import torch
import torch.nn as nn
from typing import List


ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class BranchNet(nn.Module):
    """
    MLP branch network.

    Args:
        m: number of sensor points (input dimension)
        hidden_dims: list of hidden layer sizes
        p: output dimension (must match trunk net output)
        activation: activation function name
    """

    def __init__(
        self,
        m: int,
        hidden_dims: List[int],
        p: int,
        activation: str = "tanh",
    ):
        super().__init__()

        act_cls = ACTIVATIONS[activation]
        dims = [m] + hidden_dims + [p]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no activation on last layer
                layers.append(act_cls())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, m) — input function at sensor points
        Returns:
            (batch, p)
        """
        return self.net(u)

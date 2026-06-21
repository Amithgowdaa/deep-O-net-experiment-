"""
Trunk network: encodes the query location y.

Input:  query point y (could be (x,t) for PDEs, just x here) → (batch, d_y)
Output: vector of size p → (batch, p)

The trunk net learns basis functions evaluated at y.
The branch net learns the coefficients for those basis functions.
Their dot product = the operator output at y.
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


class TrunkNet(nn.Module):
    """
    MLP trunk network.

    Args:
        d_y: dimension of query point (1 for antiderivative, 2 for (x,t) PDEs)
        hidden_dims: list of hidden layer sizes
        p: output dimension (must match branch net output)
        activation: activation function name
    """

    def __init__(
        self,
        d_y: int,
        hidden_dims: List[int],
        p: int,
        activation: str = "tanh",
    ):
        super().__init__()

        act_cls = ACTIVATIONS[activation]
        dims = [d_y] + hidden_dims + [p]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
            else:
                # Final layer gets activation too in trunk (unlike branch)
                # Reason: trunk outputs basis functions — they should be nonlinear.
                # Branch outputs coefficients — linear combination is fine.
                layers.append(act_cls())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (batch, d_y) — query locations
        Returns:
            (batch, p)
        """
        return self.net(y)

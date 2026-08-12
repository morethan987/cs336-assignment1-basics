from math import sqrt

import einx
import torch
from torch import nn


class Linear(torch.nn.Module):
    """
    Linear layer of Transformer, default to not implement bias
    Weight has a shape of (out, int) for two reasons:
        1. PyTorch uses row-major layout
        2. Academic publications usually write y = W @ x, where x is column-major layout and W has a shape of (out, in)
    For convenience, W follows academic traditions while x follows PyTorch layout. Forward should be y = x @ W.T
    State dict key: weight
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input
        y = x @ W.T
        """
        return einx.dot("... [in], out [in] -> ... out", x, self.weight)

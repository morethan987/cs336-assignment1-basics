import torch
from torch import nn


class Softmax(nn.Module):
    """
    Softmax layer
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        mx, _ = torch.max(x, dim=dim, keepdim=True)
        exp_x = torch.exp(x - mx)
        sum_x = torch.sum(exp_x, dim=dim, keepdim=True)
        return exp_x / sum_x

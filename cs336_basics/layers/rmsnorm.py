import einx
import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root mean square normalization.
    State dict keys: weight
    Args:
        d_model (int): Hidden dimension of the model
        eps (float = 1e-5): Epsilon value for numerical stability
        device (torch.device | None = None): Device to store the parameters on
        dtype (torch.dtype | None=None): Data type of the parameters
    """

    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(self.d_model, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rrms = torch.rsqrt(torch.divide(einx.dot("... in, ... in -> ...", x, x), self.d_model) + self.eps)
        res = einx.multiply("... in, in, ... -> ... in", x, self.weight, rrms)
        return res.to(in_dtype)

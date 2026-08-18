import einx
import torch
from torch import nn

from .linear import Linear


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network: SwiGLU(x) = w2 @ (SiLU(w1 @ x) * w3 @ x)
    State dict keys: w1.weight, w2.weight, w3.weight
    Args:
        d_model (int): Hidden dimension of the model
        d_ff (int | None = None): Dimensionality of the position-wise feed-forward inner layer
        device (torch.device | None = None): Device to store the parameters on
        dtype (torch.dtype | None=None): Data type of the parameters
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = self._default_d_ff() if d_ff is None else d_ff

        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(self.d_model, self.d_ff, **factory_kwargs)
        self.w2 = Linear(self.d_ff, self.d_model, **factory_kwargs)
        self.w3 = Linear(self.d_model, self.d_ff, **factory_kwargs)

    def _default_d_ff(self) -> int:
        """
        Set default d_ff value.
        d_ff is usually 4x d_model, to balance the parameter amount d_ff should be 2/3 since there are 3 matrixs in GLU.
        d_ff is ceilling to to multiple of 64 for better hardware utilization.
        """
        return int((self.d_model * 8 / 3 + 63) // 64 * 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1.forward(x)
        silu = einx.multiply("... d_ff, ... d_ff -> ... d_ff", x1, torch.sigmoid(x1))
        return self.w2.forward(einx.multiply("... d_ff, ... d_ff -> ... d_ff", silu, self.w3.forward(x)))

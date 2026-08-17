from functools import cache

import einx
import torch
from torch import nn


class RoPE(nn.Module):
    """
    Applies RoPE to the input tensor
    Args:
        theta (float): theta value for RoPE
        d_k (int): dimension of query and key vectors
        max_seq_len (int): maximum sequence length that will be input
        device (torch.device | None = None): device to store the buffer on
        dtype (torch.dtype | None=None): Data type of the parameters
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "query/key vector dimensions should be even"
        self.theta = theta
        self.d_k = d_k
        self.num_pairs = self.d_k // 2
        self.max_seq_len = max_seq_len
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self._fill_buffers()

    def _fill_buffers(self) -> None:
        """
        Fill sin and cos matrixes
        """
        pos = torch.arange(0, self.max_seq_len, **self.factory_kwargs)
        exp = -1 * torch.arange(0, self.d_k, step=2, **self.factory_kwargs) / self.d_k
        freq = self.theta**exp
        angles = einx.multiply("mx_seq, num_pairs -> mx_seq num_pairs", pos, freq)
        self.register_buffer(name="sins", tensor=torch.sin(angles), persistent=False)
        self.register_buffer(name="coss", tensor=torch.cos(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x (Float[Tensor, "... seq_len d_k"]): input tensor
        token_positions (Int[Tensor, "... seq_len"]): token positions
        """
        sins = einx.get_at("[mx_seq] num_pairs, ... seq_len -> ... seq_len num_pairs 1", self.sins, token_positions)
        coss = einx.get_at("[mx_seq] num_pairs, ... seq_len -> ... seq_len num_pairs 1", self.coss, token_positions)
        x_reshaped = einx.id("... (num_pairs k) -> ... num_pairs k", x, k=2)
        x_even, x_odd = torch.unbind(x_reshaped, -1)
        x_rev = einx.id("... num_pairs, ... num_pairs -> ... num_pairs (1 + 1)", -x_odd, x_even)
        return einx.id("... num_pairs k -> ... (num_pairs k)", x_reshaped * coss + x_rev * sins, k=2)


@cache
def get_rope(
    theta: float,
    d_k: int,
    max_seq_len: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> RoPE:
    return RoPE(theta, d_k, max_seq_len, device, dtype)

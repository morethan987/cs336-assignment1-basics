import torch
from torch import nn

from .multihead_self_attention import MultiheadSelfAttention
from .rmsnorm import RMSNorm
from .silu import SiLU


class TransformerBlock(nn.Module):
    """
    Transformer block
    State dict keys: rms1.weight, attn.q.weight, attn.k.weight, attn.v.weight, attn.o.weight, rms2.weight, ffn.w1.weight, ffn.w2.weight, ffn.w3.weight
    Args:
        d_model (int): dimensionality of the Transformer block inputs
        num_heads (int): number of heads to use in multi-head self-attention
        max_seq_len (int): maximum sequence length that will be input
        d_ff (int | None = None): dimensionality of the position-wise feed-forward inner layer
        theta (float | None = None): theta value for RoPE, if not passed will not apply RoPE
        device (torch.device | None = None): device to store the parameters on
        dtype (torch.dtype | None=None): data type of the parameters
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        d_ff: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        factory_kwargs = {"device": device, "dtype": dtype}
        self.rms1 = RMSNorm(d_model, **factory_kwargs)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, **factory_kwargs)
        self.rms2 = RMSNorm(d_model, **factory_kwargs)
        self.ffn = SiLU(d_model, d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x1 = self.rms1(x)
        x2 = self.attn(x1, token_positions)
        x3 = x + x2
        x4 = self.rms2(x3)
        x5 = self.ffn(x4)
        return x3 + x5

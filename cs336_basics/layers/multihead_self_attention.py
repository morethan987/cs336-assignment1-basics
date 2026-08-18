import einx
import torch
from torch import nn

from .linear import Linear
from .rope import RoPE
from .utils import scaled_dot_product_attention


class MultiheadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention
    State dict keys: q.weight, k.weight, v.weight, o.weight
    Args:
        d_model (int): dimensionality of the Transformer block inputs
        num_heads (int): number of heads to use in multi-head self-attention
        theta (float | None = None): theta value for RoPE, if not passed will not apply RoPE
        max_seq_len (int | None = None): maximum sequence length that will be input, if not passed will not apply RoPE
        device (torch.device | None = None): device to store the buffer on
        dtype (torch.dtype | None=None): data type of the parameters
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "Token embedding dim should be the multiple of num_heads"
        self.d_kv = d_model // num_heads  # standard default value
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.rope = None
        if self.theta and self.max_seq_len:
            self.rope = RoPE(self.theta, self.d_kv, self.max_seq_len, **self.factory_kwargs)
        if (self.theta and not self.max_seq_len) or (not self.theta and self.max_seq_len):
            raise ValueError(
                f"theta or max_seq_len should not be None."
                f"current theta:{self.theta}"
                f"current max_seq_len: {self.max_seq_len}"
            )

        self.q = Linear(self.d_model, self.d_model, **self.factory_kwargs)
        self.k = Linear(self.d_model, self.d_model, **self.factory_kwargs)
        self.v = Linear(self.d_model, self.d_model, **self.factory_kwargs)
        self.o = Linear(self.d_model, self.d_model, **self.factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        # heads split
        Q = einx.id("... seq_len (h d_kv) -> ... h seq_len d_kv", Q, h=self.num_heads)
        K = einx.id("... seq_len (h d_kv) -> ... h seq_len d_kv", K, h=self.num_heads)
        V = einx.id("... seq_len (h d_kv) -> ... h seq_len d_kv", V, h=self.num_heads)
        seq_len = x.shape[-2]
        if self.rope:  # apply RoPE
            if token_positions is None:
                raise ValueError("RoPE is set but token_positions is None. Pass token position tensor for RoPE!")
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
        att = scaled_dot_product_attention(Q, K, V, mask)
        return self.o(einx.id("... h seq_len d_kv -> ... seq_len (h d_kv)", att))

import torch
from torch import nn

from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """
    Transformer language model, return unnormalized prediction
    State dict keys:
    embed.weight, output_rms.weight, output_linear.weight
    transformers.{id}.rms1.weight, transformers.{id}.attn.q.weight, transformers.{id}.attn.k.weight, transformers.{id}.attn.v.weight, transformers.{id}.attn.o.weight, transformers.{id}.rms2.weight, transformers.{id}.ffn.w1.weight, transformers.{id}.ffn.w2.weight, transformers.{id}.ffn.w3.weight
    Args:
        vocab_size (int): The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix
        context_length (int): The maximum context length, necessary for determining the dimensionality of the RoPE sin and cos buffer
        num_layers (int): The number of Transformer blocks to use
        d_model (int): dimensionality of the Transformer block inputs
        num_heads (int): number of heads to use in multi-head self-attention
        d_ff (int | None = None): dimensionality of the position-wise feed-forward inner layer
        rope_theta (float | None = None): theta value for RoPE, if not passed will not apply RoPE
        device (torch.device | None = None): device to store the parameters on
        dtype (torch.dtype | None=None): data type of the parameters
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        rope_theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.embed = Embedding(vocab_size, d_model, **self.factory_kwargs)
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, **self.factory_kwargs)
                for _ in range(self.num_layers)
            ]
        )
        self.output_rms = RMSNorm(d_model, **self.factory_kwargs)
        self.output_linear = Linear(d_model, self.vocab_size, **self.factory_kwargs)

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(
                f"Input sequence is too long:\nseq_len = {seq_len} > max_context_length = {self.context_length}"
            )
        if token_positions is None:
            token_positions = torch.arange(0, seq_len, dtype=torch.int, device=token_ids.device)
        x = self.embed(token_ids)
        for block in self.transformers:
            x = block(x, token_positions)
        x = self.output_rms(x)
        return self.output_linear(x)

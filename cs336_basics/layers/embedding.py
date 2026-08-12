import einx
import torch
from torch import nn


class Embedding(nn.Module):
    """
    Embedding layer of Transformer, maps ids to vector
    State dict keys: weight
    Args:
        num_embeddings (int): Size of the vocabulary
        embedding_dim (int): Dimension of the embedding vectors
        device (torch.device | None = None): Device to store the parameters on
        dtype (torch.dtype | None = None): Data type of the parameters
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return einx.get_at("[n] d, ... select -> ... select d", self.weight, token_ids)

import einx
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    mx, _ = torch.max(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - mx)
    sum_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_x


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = torch.tensor(Q.shape[-1])
    # n and m all respresent seq_len, only to tag matrix shape: (n, m) or (m, n)
    scaled_dot = torch.multiply(torch.rsqrt(d_k), einx.dot("... n [d_k], ... m [d_k] -> ... n m", Q, K))
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(~mask, float("-inf"))
    return einx.dot("... n [m], ... [m] d_v -> ... n d_v", softmax(scaled_dot, -1), V)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mx = einx.max("... ([vocab_size])", inputs)
    inputs_shifted = inputs - mx
    log_sum_exp = torch.log(einx.sum("... [vocab_size]", torch.exp(inputs_shifted)))
    target_logits = einx.get_at("... [vocab_size], ... -> ...", inputs_shifted, targets)
    return torch.mean(log_sum_exp - target_logits)

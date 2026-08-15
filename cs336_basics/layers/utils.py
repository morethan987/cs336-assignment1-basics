import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    mx, _ = torch.max(x, dim=dim, keepdim=True)
    exp_x = torch.exp(x - mx)
    sum_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_x

import math
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import torch


def cosine_annealing(t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int) -> float:
    """
    Cosine annealing learning rate scheduling
    Args:
        t (int): current iteration
        alpha_max (float): maximum learning rate
        alpha_min (float): minimum (final) learning rate
        t_w (int): number of warmup iterations
        t_c (int): final iteration of cosine annealing
    Return:
        lr_t (float): learning rate at iteration t
    """
    if t < t_w:  # warmup
        return (alpha_max / t_w) * t
    elif t <= t_c:  # cosine annealing
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi / (t_c - t_w) * (t - t_w)))
    else:
        return alpha_min


@torch.no_grad()
def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return

    total_norm = torch.stack([g.norm() for g in grads]).norm()
    clip_coef = torch.clamp(max_l2_norm / (eps + total_norm), max=1.0)
    for g in grads:
        g.mul_(clip_coef)


def load_data(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input sequences and next-token targets from tokenized data.
    Args:
        dataset: 1D numpy array of token IDs.
        batch_size (B): Number of sequences to sample in a batch.
        context_length (m): Length of each sequence.
        device: PyTorch device (e.g., 'cpu', 'cuda:0', 'mps').
    Returns:
        inputs: Tensor of shape (batch_size, context_length) on device.
        targets: Tensor of shape (batch_size, context_length) on device.
    """
    max_id = len(dataset) - context_length
    starts = np.random.randint(0, max_id, size=batch_size)

    inputs_list = [dataset[i : i + context_length] for i in starts]
    targets_list = [dataset[i + 1 : i + 1 + context_length] for i in starts]

    inputs = torch.tensor(np.array(inputs_list), dtype=torch.long, device=device)
    targets = torch.tensor(np.array(targets_list), dtype=torch.long, device=device)
    return inputs, targets

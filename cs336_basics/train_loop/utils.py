import math
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import IO, BinaryIO

import einx
import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.layers import TransformerLM, softmax


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
def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """In-place change gradients and return original norm"""
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return torch.Tensor(0.0)

    total_norm = torch.stack([g.norm() for g in grads]).norm()
    clip_coef = torch.clamp(max_l2_norm / (eps + total_norm), max=1.0)
    for g in grads:
        g.mul_(clip_coef)
    return total_norm


def load_data(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    generator: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input sequences and next-token targets from tokenized data.
    Args:
        dataset: 1D numpy array of token IDs.
        batch_size (B): Number of sequences to sample in a batch.
        context_length (m): Length of each sequence.
        device: PyTorch device (e.g., 'cpu', 'cuda:0', 'mps').
        generator: np.random.Generator instance
    Returns:
        inputs: Tensor of shape (batch_size, context_length) on device.
        targets: Tensor of shape (batch_size, context_length) on device.
    """
    max_id = len(dataset) - context_length
    starts = generator.integers(0, max_id, size=batch_size)

    inputs_array = np.stack([dataset[i : i + context_length] for i in starts])
    targets_array = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])

    inputs = torch.tensor(inputs_array, dtype=torch.long, device=device)
    targets = torch.tensor(targets_array, dtype=torch.long, device=device)
    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Dump all the state from the model, optimizer and iteration into the file-like object out
    Uses atomic writes if 'out' is a file path to prevent checkpoint corruption.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    if isinstance(out, (str, os.PathLike)):  # out is a path
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = out_path.with_name(f".{out_path.name}.tmp.{uuid.uuid4().hex[:8]}")

        try:
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, out_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    else:  # out is stream or file object
        torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load a checkpoint from src (path or file-like object), and then recover the model and optimizer states from that checkpoint
    Handles cross-device mapping and supports weights_only for security.
    Returns:
        iteration (int): the previously-serialized number of iterations.
    """
    try:
        target_device = next(model.parameters()).device
    except StopIteration:
        target_device = torch.device("cpu")

    checkpoint = torch.load(src, map_location=target_device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]


def parse_dtype(dtype_str: str) -> torch.dtype:
    """convert string to torch.dtype"""
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    dtype_str = dtype_str.lower()
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    raise ValueError(f"Unsupported dtype: '{dtype_str}'. Choose from {list(dtype_map.keys())}")


@torch.inference_mode()
def decoding(
    x: torch.Tensor,
    model: TransformerLM,
    eos_id: int,
    temp: float,
    p: float,
    max_generation: int | None = None,
) -> torch.Tensor:
    """
    Apply temperature scaling and Top-p sampling
    Args:
        x (torch.Tensor): tokenized input prompts, shape (... batch_size, seq_len)
        model (TransformerLM): transformer lm instance
        eos_id (int): int id for end of file token
        temp (float): temperature value
        p (float): threshold value, select from (0, 1)
        max_generation (int): max length to generate
    Return:
        output (torch.Tensor): generated token ids
    """
    prompt_length = x.shape[-1]
    left_budget = model.context_length - prompt_length
    if max_generation is None or max_generation > left_budget:
        max_generation = left_budget
    if max_generation <= 0:
        raise ValueError(
            f"Max generation length is negative! "
            f"Max context length is {model.context_length} "
            f"Prompt length is {prompt_length}"
            f"Current max_generation is {max_generation}"
        )

    batch_shape = x.shape[:-1]
    finished = torch.zeros(batch_shape, dtype=torch.bool, device=x.device)
    generated = x[..., :0]

    for _ in range(max_generation):
        logits = model(x)  # (... batch_size, seq_len, vocab_size)
        last_logits: torch.Tensor = einx.get_at("... [s] v, -> ... v", logits, -1)
        t_scaled = softmax(last_logits.div_(temp), dim=-1)

        sorted_prob, idxes = torch.sort(t_scaled, descending=True)
        csum = torch.cumsum(sorted_prob, dim=-1).sub_(sorted_prob)
        mask = csum.greater_equal(p)
        sorted_prob.masked_fill_(mask, 0.0)

        # sampling
        sample = torch.multinomial(sorted_prob, 1)
        next_token = einx.get_at("... [v], ... 1 -> ... 1", idxes, sample)

        # padding with eos
        next_token = einx.where("..., , ... 1 -> ... 1", finished, eos_id, next_token)

        # cat
        generated = einx.id("... a, ... b -> ... (a + b)", generated, next_token)
        x = einx.id("... a, ... b -> ... (a + b)", x, next_token)

        is_eos = einx.id("... 1 -> ...", next_token) == eos_id
        finished = finished | is_eos
        if finished.all():
            break

    return generated

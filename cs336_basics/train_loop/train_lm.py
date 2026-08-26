import argparse
import json
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime as dt
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import wandb
from torch import nn

from cs336_basics.layers import TransformerLM, cross_entropy

from .adamw import AdamW
from .utils import cosine_annealing, gradient_clipping, load_data, parse_dtype, save_checkpoint


def main(args: Namespace):
    # timestamp and paths
    timestamp = dt.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_{timestamp}" if args.name else f"run_{timestamp}"
    run_dir = Path(args.checkpoints_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = run_dir / "checkpoint_latest.pt"
    save_hyperparams(args, run_dir)

    # wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            dir=str(run_dir),
        )

    # set random seed
    numpy_rng = np.random.default_rng(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    # model and optimizer
    lm = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        args.device,
        args.dtype,
    )
    opt = AdamW(lm.parameters(), args.lr, args.weight_decay, args.eps, tuple(args.betas))

    # lazily load data set. Hardcoded dtype as uint16, refer to `scripts/encode_datasets.py`
    dataset = np.memmap(args.train_data, mode="r", dtype=np.uint16)
    val_data = None
    if args.valid_data is not None:
        val_data = np.memmap(args.valid_data, mode="r", dtype=np.uint16)

    for step in range(1, args.max_steps + 1):
        # dynamic lr
        lr_t = cosine_annealing(
            t=step,
            alpha_max=args.lr,
            alpha_min=args.min_lr,
            t_w=args.warmup_steps,
            t_c=args.max_steps,
        )
        for param_group in opt.param_groups:
            param_group["lr"] = lr_t

        # forward and backward
        x, targets = load_data(dataset, args.batch_size, args.context_length, args.device, numpy_rng)
        opt.zero_grad()
        y = lm(x)
        loss = cross_entropy(y, targets)
        loss.backward()
        original_norm = gradient_clipping(lm.parameters(), max_l2_norm=args.grad_clip)
        opt.step()

        # log and checkpoint
        if (step) % args.log_interval == 0:
            print(f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f} | LR: {lr_t:.6e}")
            if args.use_wandb:
                log_data = {
                    "train/loss": loss.item(),
                    "train/learning_rate": lr_t,
                    "train/step": step,
                    "train/original_grad_norm": original_norm.item(),
                }
                wandb.log(log_data, step=step)

        if (step) % args.save_interval == 0:
            step_ckpt_name = f"checkpoint_step_{step}.pt"
            step_ckpt_path = run_dir / step_ckpt_name
            save_checkpoint(lm, opt, step, step_ckpt_path)
            # save a latest checkpoint for convenience via symlink
            _update_symlink(latest_ckpt_path, step_ckpt_name)

        if val_data is not None and step % args.val_interval == 0:
            val_loss = valid(args, lm, val_data, args.max_val_batches)
            print(f"Step {step}/{args.max_steps} | Valid Loss: {val_loss:.4f}")
            if args.use_wandb:
                log_data = {
                    "valid/loss": val_loss,
                    "valid/step": step,
                }
                wandb.log(log_data, step=step)

    print("Training complete!")
    max_ckpt_name = f"checkpoint_step_{args.max_steps}.pt"
    max_ckpt_path = run_dir / max_ckpt_name
    if not max_ckpt_path.exists():
        save_checkpoint(lm, opt, args.max_steps, max_ckpt_path)
        _update_symlink(latest_ckpt_path, max_ckpt_name)


@contextmanager
def eval_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


@torch.inference_mode()
def valid(args: Namespace, model: nn.Module, data: npt.NDArray, max_val_batches: int | None = None) -> float:
    """
    Validate on valid_dataset without stochastic sampling
    """
    with eval_mode(model):
        total_samples = (len(data) - 1) // args.context_length
        num_batches = total_samples // args.batch_size

        if max_val_batches is not None:
            num_batches = min(num_batches, max_val_batches)

        if num_batches == 0:
            print("[Warning] Validation dataset is too small for the given batch_size and context_length.")
            return float("nan")

        batch_tokens = args.batch_size * args.context_length
        total_loss = 0.0

        for i in range(num_batches):
            start = i * batch_tokens
            end = start + batch_tokens

            x_np = data[start:end]
            y_np = data[start + 1 : end + 1]

            x = torch.from_numpy(x_np).view(args.batch_size, args.context_length).to(args.device, dtype=torch.long)
            targets = (
                torch.from_numpy(y_np).view(args.batch_size, args.context_length).to(args.device, dtype=torch.long)
            )

            # forward and loss
            logits = model(x)
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()

        val_loss = total_loss / num_batches
        return val_loss


def _update_symlink(ckpt: Path, relative_name: str):
    if ckpt.is_symlink() or ckpt.exists():
        ckpt.unlink()
    ckpt.symlink_to(relative_name)


def save_hyperparams(args: Namespace, save_dir: Path):
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        args_dict = {k: str(v) if isinstance(v, (torch.dtype, Path)) else v for k, v in vars(args).items()}
        json.dump(args_dict, f, indent=4)


def args_validate(args: Namespace):
    # check missing params
    required_args = [
        "vocab_size",
        "context_length",
        "num_layers",
        "d_model",
        "num_heads",
        "d_ff",
        "train_data",
        "lr",
        "max_steps",
        "batch_size",
    ]
    for param in required_args:
        if getattr(args, param) is None:
            raise ValueError(f"Missing required argument: --{param}")

    # 2. Transformer
    if args.d_model % args.num_heads != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads})")

    # 3. data path
    if not Path(args.train_data).is_file():
        raise FileNotFoundError(f"Training data path does not exist: {args.train_data}")
    if args.valid_data and not Path(args.valid_data).is_file():
        raise FileNotFoundError(f"Validation data path does not exist: {args.valid_data}")

    # 4. lr and scheduler
    if args.min_lr > args.lr:
        raise ValueError(f"min_lr ({args.min_lr}) cannot be greater than base lr ({args.lr})")
    if args.warmup_steps >= args.max_steps:
        raise ValueError(f"warmup_steps ({args.warmup_steps}) cannot be greater than max_steps ({args.max_steps})")

    # 5. device
    if args.device and "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device specified as '{args.device}', but CUDA is not available.")


def args_parse() -> Namespace:
    parser = argparse.ArgumentParser(description="Training script for Transformer")

    # transformer params
    parser.add_argument("--vocab_size", type=int, help="The size of vocabulary")
    parser.add_argument("--context_length", type=int, help="The context length for training")
    parser.add_argument("--num_layers", type=int, help="The number of transformer blocks")
    parser.add_argument("--d_model", type=int, help="The hidden dimension of the LM")
    parser.add_argument("--num_heads", type=int, help="The number of heads for multi-head self-attention")
    parser.add_argument("--d_ff", type=int, help="The dimension of feed-forward layer")
    parser.add_argument("--rope_theta", type=float, help="The theta for RoPE")
    parser.add_argument("--device", type=str, help="The device string such as 'cuda:0'")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16, help="Data type for training")

    # training params
    parser.add_argument("--name", type=str, default="experiment", help="Name of the traning run")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Root directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--log_interval", type=int, default=50, help="Save log every N steps")
    parser.add_argument("--train_data", type=str, help="Path to tokenized training data")
    parser.add_argument("--max_steps", type=int, help="Max steps/iterations to train")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for AdamW")
    parser.add_argument("--eps", type=float, default=1e-8, help="Small value for numerical stability")
    parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.95), help="Params for AdamW to update momentum")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm for clipping (default 1.0)")
    parser.add_argument("--numpy_seed", type=int, default=42, help="Random seed for numpy data loader")
    parser.add_argument("--torch_seed", type=int, default=45, help="Random seed for torch")

    # validating params
    parser.add_argument("--valid_data", type=str, help="Path to tokenized validating data")
    parser.add_argument("--val_interval", type=int, default=200, help="Validate on valid_data every N steps")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Max validate batches")

    # lr scheduling
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum/final learning rate for cosine schedule")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of linear warmup iterations")

    # wandb params
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-basics", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/username/team")

    args = parser.parse_args()
    args_validate(args)
    return args


if __name__ == "__main__":
    main(args_parse())

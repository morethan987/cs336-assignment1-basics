import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime as dt
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import wandb

from cs336_basics.layers import TransformerLM, cross_entropy

from .adamw import AdamW
from .utils import cosine_annealing, gradient_clipping, load_data, parse_dtype, save_checkpoint


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int | None
    rope_theta: float | None = 10000.0
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: torch.dtype = torch.bfloat16

    def validate(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device specified as '{self.device}', but CUDA is not available.")


@dataclass
class TrainConfig:
    # path and tags
    name: str = "experiment"
    description: str = ""
    checkpoints_dir: Path = Path("checkpoints")
    train_data: Path = Path()
    valid_data: Path | None = None

    # hyperparams
    max_steps: int = 10000
    batch_size: int = 64
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    eps: float = 1e-8
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # schedule and interval
    save_interval: int = 200
    log_interval: int = 50
    val_interval: int = 200
    max_val_batches: int | None = None

    # random seeds
    numpy_seed: int = 42
    torch_seed: int = 45

    # wandb log
    use_wandb: bool = False
    wandb_project: str = "cs336-basics"
    wandb_entity: str | None = None

    def validate(self):
        if not self.train_data.is_file():
            raise FileNotFoundError(f"Training data not found: {self.train_data}")
        if self.valid_data and not self.valid_data.is_file():
            raise FileNotFoundError(f"Validation data not found: {self.valid_data}")
        if self.min_lr > self.lr:
            raise ValueError(f"min_lr ({self.min_lr}) > lr ({self.lr})")
        if self.warmup_steps >= self.max_steps:
            raise ValueError(f"warmup_steps ({self.warmup_steps}) >= max_steps ({self.max_steps})")


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_cfg = cfg.model
        self.train_cfg = cfg.train

        # set seeds and init directories
        self._set_seed()
        self.run_dir = self._setup_run_dir()

        # wandb init
        if self.train_cfg.use_wandb:
            wandb.init(
                project=self.train_cfg.wandb_project,
                entity=self.train_cfg.wandb_entity,
                name=self.run_dir.name,
                config=asdict(self.cfg),
                dir=str(self.run_dir),
            )

        # construct model and optimizer
        self.model = TransformerLM(
            self.model_cfg.vocab_size,
            self.model_cfg.context_length,
            self.model_cfg.num_layers,
            self.model_cfg.d_model,
            self.model_cfg.num_heads,
            self.model_cfg.d_ff,
            self.model_cfg.rope_theta,
            self.model_cfg.device,
            self.model_cfg.dtype,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
            eps=self.train_cfg.eps,
            betas=self.train_cfg.betas,
        )

        # load data
        self.train_dataset = np.memmap(self.train_cfg.train_data, mode="r", dtype=np.uint16)
        self.val_dataset = (  # hardcoded uint16, refer to `scripts/encode_datasets.py`
            np.memmap(self.train_cfg.valid_data, mode="r", dtype=np.uint16) if self.train_cfg.valid_data else None
        )

    def _set_seed(self):
        self.numpy_rng = np.random.default_rng(self.train_cfg.numpy_seed)
        torch.manual_seed(self.train_cfg.torch_seed)

    def _setup_run_dir(self) -> Path:
        timestamp = dt.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.train_cfg.name}_{timestamp}"
        run_dir = self.train_cfg.checkpoints_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # save hyperparams
        with open(run_dir / "args.json", "w", encoding="utf-8") as f:

            def serialize(obj):
                if isinstance(obj, (torch.device, torch.dtype, Path)):
                    return str(obj)
                raise TypeError(f"Type {type(obj)} not serializable")

            json.dump(asdict(self.cfg), f, indent=4, default=serialize)
        return run_dir

    def _get_lr(self, step: int) -> float:
        return cosine_annealing(
            t=step,
            alpha_max=self.train_cfg.lr,
            alpha_min=self.train_cfg.min_lr,
            t_w=self.train_cfg.warmup_steps,
            t_c=self.train_cfg.max_steps,
        )

    def train_step(self, step: int) -> tuple[float, torch.Tensor, torch.Tensor]:
        lr_t = self._get_lr(step)
        self.optimizer.set_lr(lr_t)

        # take data
        x, targets = load_data(
            self.train_dataset,
            self.train_cfg.batch_size,
            self.model_cfg.context_length,
            self.model_cfg.device,
            self.numpy_rng,
        )

        # forward and backward
        self.optimizer.zero_grad()
        logits = self.model(x, torch.arange(0, x.shape[-1], dtype=torch.int, device=x.device))
        loss = cross_entropy(logits, targets)
        loss.backward()

        # gradient clipping and update
        grad_norm = gradient_clipping(self.model.parameters(), max_l2_norm=self.train_cfg.grad_clip)
        self.optimizer.step()
        return lr_t, loss, grad_norm

    @torch.inference_mode()
    def evaluate(self) -> float:
        if self.val_dataset is None:
            return float("nan")
        self.model.eval()

        bs = self.train_cfg.batch_size
        ctx_len = self.model_cfg.context_length
        batch_tokens = bs * ctx_len
        total_batches = (len(self.val_dataset) - 1) // batch_tokens
        if self.train_cfg.max_val_batches:
            total_batches = min(total_batches, self.train_cfg.max_val_batches)
        if total_batches == 0:
            return float("nan")

        total_loss = 0.0
        for i in range(total_batches):
            start = i * batch_tokens
            end = start + batch_tokens

            x = (
                torch.from_numpy(self.val_dataset[start:end].copy())
                .view(bs, ctx_len)
                .to(self.model_cfg.device, dtype=torch.long)
            )
            targets = (
                torch.from_numpy(self.val_dataset[start + 1 : end + 1].copy())
                .view(bs, ctx_len)
                .to(self.model_cfg.device, dtype=torch.long)
            )

            # forward and loss
            logits = self.model(x, torch.arange(0, x.shape[-1], dtype=torch.int, device=x.device))
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()

        self.model.train()
        return total_loss / total_batches

    def save_checkpoint(self, step: int):
        ckpt_name = f"checkpoint_step_{step}.pt"
        ckpt_path = self.run_dir / ckpt_name
        save_checkpoint(self.model, self.optimizer, step, ckpt_path)

        # update latest symlink
        latest_path = self.run_dir / "checkpoint_latest.pt"
        if latest_path.is_symlink() or latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(ckpt_name)

    def fit(self):
        print(f"Starting training run: {self.run_dir.name}")
        print(self.train_cfg.description)

        tokens_per_step = self.train_cfg.batch_size * self.model_cfg.context_length
        start_time = time.perf_counter()

        if self.val_dataset is not None:
            init_val_loss = self.evaluate()
            print(f"[Init] Step 0/{self.train_cfg.max_steps} | Initial Valid Loss: {init_val_loss:.4f}")
            if self.train_cfg.use_wandb:
                wandb.log({"valid/loss": init_val_loss}, step=0)

        self.model.train()
        train_time_accum = 0.0
        steps_since_log = 0

        for step in range(1, self.train_cfg.max_steps + 1):
            is_last_step = step == self.train_cfg.max_steps

            step_start = time.perf_counter()
            lr, loss, grad_norm = self.train_step(step)
            train_time_accum += time.perf_counter() - step_start
            steps_since_log += 1

            if step % self.train_cfg.log_interval == 0 or is_last_step:
                current_time = time.perf_counter()
                elapsed_total = current_time - start_time
                step_time_ms = (train_time_accum / steps_since_log) * 1000
                tok_per_sec = (tokens_per_step * steps_since_log) / train_time_accum

                print(
                    f"[{dt.now(timezone(timedelta(hours=8))).strftime('%H:%M:%S')}] "
                    f"Step {step}/{self.train_cfg.max_steps} | "
                    f"Loss: {loss:.4f} | "
                    f"LR: {lr:.6e} | "
                    f"Step: {step_time_ms:.1f}ms | "
                    f"Throughput: {tok_per_sec:.0f} tok/s | "
                    f"Elapsed: {timedelta(seconds=int(elapsed_total))}"
                )

                if self.train_cfg.use_wandb:
                    wandb.log(
                        {
                            "train/learning_rate": lr,
                            "train/loss": loss.item(),
                            "train/original_grad_norm": grad_norm.item(),
                            "perf/step_time_ms": step_time_ms,
                            "perf/tokens_per_second": tok_per_sec,
                            "perf/wallclock_time_s": elapsed_total,
                        },
                        step=step,
                    )

                train_time_accum = 0.0
                steps_since_log = 0

            if self.val_dataset is not None and (step % self.train_cfg.val_interval == 0 or is_last_step):
                val_loss = self.evaluate()
                print(f"Step {step}/{self.train_cfg.max_steps} | Valid Loss: {val_loss:.4f}")
                if self.train_cfg.use_wandb:
                    wandb.log({"valid/loss": val_loss}, step=step)
                self.model.train()

            if step % self.train_cfg.save_interval == 0 or is_last_step:
                self.save_checkpoint(step)

        print("Training complete!")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train Transformer Language Model", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument("--vocab_size", type=int, required=True)
    model_group.add_argument("--context_length", type=int, required=True)
    model_group.add_argument("--num_layers", type=int, required=True)
    model_group.add_argument("--d_model", type=int, required=True)
    model_group.add_argument("--num_heads", type=int, required=True)
    model_group.add_argument("--d_ff", type=int, required=True)
    model_group.add_argument("--rope_theta", type=float, default=10000.0)
    model_group.add_argument(
        "--device", type=torch.device, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model_group.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)

    # Train
    train_group = parser.add_argument_group("Training Arguments")
    train_group.add_argument("--name", type=str, default="experiment")
    train_group.add_argument("--desc", type=str, default="")
    train_group.add_argument("--checkpoints_dir", type=Path, default=Path("checkpoints"))
    train_group.add_argument("--train_data", type=Path, required=True)
    train_group.add_argument("--valid_data", type=Path, default=None)
    train_group.add_argument("--max_steps", type=int, required=True)
    train_group.add_argument("--batch_size", type=int, required=True)
    train_group.add_argument("--lr", type=float, required=True)
    train_group.add_argument("--min_lr", type=float, default=1e-5)
    train_group.add_argument("--warmup_steps", type=int, default=500)
    train_group.add_argument("--weight_decay", type=float, default=0.1)
    train_group.add_argument("--eps", type=float, default=1e-8)
    train_group.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.95))
    train_group.add_argument("--grad_clip", type=float, default=1.0)
    train_group.add_argument("--save_interval", type=int, default=200)
    train_group.add_argument("--log_interval", type=int, default=50)
    train_group.add_argument("--val_interval", type=int, default=200)
    train_group.add_argument("--max_val_batches", type=int, default=None)
    train_group.add_argument("--numpy_seed", type=int, default=42)
    train_group.add_argument("--torch_seed", type=int, default=45)

    # WandB
    wandb_group = parser.add_argument_group("WandB Arguments")
    wandb_group.add_argument("--use_wandb", action="store_true")
    wandb_group.add_argument("--wandb_project", type=str, default="cs336-basics")
    wandb_group.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    # dataclass
    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=args.dtype,
    )
    train_cfg = TrainConfig(
        name=args.name,
        description=args.desc,
        checkpoints_dir=args.checkpoints_dir,
        train_data=args.train_data,
        valid_data=args.valid_data,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eps=args.eps,
        betas=tuple(args.betas),
        grad_clip=args.grad_clip,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        max_val_batches=args.max_val_batches,
        numpy_seed=args.numpy_seed,
        torch_seed=args.torch_seed,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    # validate
    model_cfg.validate()
    train_cfg.validate()

    return Config(model=model_cfg, train=train_cfg)


if __name__ == "__main__":
    config = parse_args()
    trainer = Trainer(config)
    trainer.fit()

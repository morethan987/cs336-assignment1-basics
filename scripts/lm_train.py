import argparse
from pathlib import Path

import torch

from cs336_basics.train_loop import Config, ModelConfig, TrainConfig, Trainer


def mini_test():
    """
    Test quick overfitting on mini dataset
    """
    model_cfg = ModelConfig(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
        device=torch.device("cuda:0"),
    )

    train_cfg = TrainConfig(
        name="mini_test",
        description="Test quick overfitting on mini dataset",
        train_data=Path(
            "/root/cs336/cs336-assignment1-basics/outputs/bpe_tokenizer/20260829_224820/TinyStoriesV2-GPT4-mini-tokenized.bin"
        ),
        valid_data=Path(
            "/root/cs336/cs336-assignment1-basics/outputs/bpe_tokenizer/20260829_224820/TinyStoriesV2-GPT4-mini-tokenized.bin"
        ),
        max_steps=1000,
        batch_size=16,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=50,
        weight_decay=0.01,
        eps=1e-7,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        save_interval=100,
        log_interval=5,
        val_interval=50,
        use_wandb=True,
        wandb_project="cs336_basics",
        wandb_entity="morethan987-chongqing-university",
    )

    # validate
    model_cfg.validate()
    train_cfg.validate()

    cfg = Config(model=model_cfg, train=train_cfg)
    trainer = Trainer(cfg)
    trainer.fit()


def tinystories_train(args: argparse.Namespace):
    model_cfg = ModelConfig(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
        device=torch.device("cuda:0"),
    )

    scale = int(256 / args.batch_size)
    train_cfg = TrainConfig(
        name=f"tinystories_bs_{args.batch_size}",
        description="Sweeping batch size",
        train_data=Path(
            "/root/cs336/cs336-assignment1-basics/outputs/bpe_tokenizer/20260803_173732/TinyStoriesV2-GPT4-train-tokenized.bin"
        ),
        valid_data=Path(
            "/root/cs336/cs336-assignment1-basics/outputs/bpe_tokenizer/20260803_173732/TinyStoriesV2-GPT4-valid-tokenized.bin"
        ),
        max_steps=5000 * scale,
        batch_size=args.batch_size,
        lr=1.5e-3,
        min_lr=1.5e-4,
        warmup_steps=100 * scale,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        save_interval=250 * scale,
        log_interval=20 * scale,
        val_interval=100 * scale,
        use_wandb=True,
        wandb_project="cs336_basics",
        wandb_entity="morethan987-chongqing-university",
    )

    # validate
    model_cfg.validate()
    train_cfg.validate()

    cfg = Config(model=model_cfg, train=train_cfg)
    trainer = Trainer(cfg)
    trainer.fit()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweeping batch size")
    parser.add_argument("--batch_size", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    # pueue add -l "bs=128" "uv run scripts/lm_train.py --batch_size 128"
    # mini_test()
    args = parse()
    tinystories_train(args)

from pathlib import Path

import torch

from cs336_basics.train_loop import Config, ModelConfig, TrainConfig, Trainer


def main():
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


if __name__ == "__main__":
    main()

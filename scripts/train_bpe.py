from cs336_basics.bpe_trainer import BPE_Trainer


def train_tinystories() -> None:
    """
    Train BPE on TinyStories dataset
    Start in project root with `uv run scripts/train_bpe.py`
    """
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    num_processes = 4
    trainer = BPE_Trainer(input_path, vocab_size, special_tokens, num_processes)
    trainer.train()
    vp, mp = trainer.pickling()
    print(f"successfully saved into:\n{vp}\n{mp}")


def train_expts_owt() -> None:
    """
    Train BPE on OpenWebText dataset
    Start in project root with `uv run scripts/train_bpe.py`
    """
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    num_processes = 4
    trainer = BPE_Trainer(input_path, vocab_size, special_tokens, num_processes)
    trainer.train()
    vp, mp = trainer.pickling()
    print(f"successfully saved into:\n{vp}\n{mp}")


if __name__ == "__main__":
    # train_tinystories()
    train_expts_owt()

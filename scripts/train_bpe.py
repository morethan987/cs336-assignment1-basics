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


def visualize(v_path: str, m_path: str) -> None:
    vp, mp = BPE_Trainer.unpickling(v_path, m_path)

    print("vocabulary list length: ", len(vp))
    print("special tokens: ", vp[256])
    print("first 20 tokens: ")
    vls = list(vp.values())
    print(vls[257:277])
    print("last 20 tokens: ")
    print(vls[-20:])
    print("longest token: ")
    print(max(vls, key=lambda x: len(x)).decode("utf-8"))


if __name__ == "__main__":
    train_tinystories()
    # train_expts_owt()
    
    # v_path = "outputs/bpe_trainer/20260725_204945/vocab.pkl"
    # m_path = "outputs/bpe_trainer/20260725_204945/merges.pkl"
    # visualize(v_path, m_path)
    # v_path = "outputs/bpe_trainer/20260727_015403/vocab.pkl"
    # m_path = "outputs/bpe_trainer/20260727_015403/merges.pkl"
    # visualize(v_path, m_path)

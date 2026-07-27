from cs336_basics.bpe_trainer import BPE_Trainer
from scripts.mem_tracker import MemoryTracker

trainer = BPE_Trainer("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"], 4)

mt = MemoryTracker(interval=0.05)
mt.start()
trainer.train()
mt.stop()

import time, cProfile, pstats
from io import StringIO
from cs336_basics.bpe_trainer import BPE_Trainer

trainer = BPE_Trainer(
    "data/TinyStoriesV2-GPT4-train.txt",
    10000,
    ["<|endoftext|>"],
    4,
)

# --- 模式 A:只测真实墙钟(用这个数当"真实训练时间") ---
# t0 = time.perf_counter()
# trainer.train()
# print(f"wall-clock train(): {time.perf_counter() - t0:.2f}s")

# --- 模式 B:cProfile 定位热点 ---
pr = cProfile.Profile()
pr.enable()
trainer.train()
pr.disable()
s = StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
print(s.getvalue())

import time
import threading
import statistics
import psutil


_active = None


def stage(name: str) -> None:
    """临时:在 train() 阶段边界调用,给样本打标签。测完连 import 一起删。"""
    if _active is not None:
        _active.set_stage(name)


class MemoryTracker:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._stage = "init"
        self._lock = threading.Lock()
        self._samples: list[tuple[float, str, int]] = []
        self._t0 = 0.0
        self._thread = None
        self._stop = threading.Event()

    def set_stage(self, name: str) -> None:
        with self._lock:
            self._stage = name

    def _total_rss(self) -> int:
        # 主进程 + 所有子孙进程(worker)RSS 求和 —— 这样 pretokenize 的 4 个 worker 才不会被漏掉
        proc = psutil.Process()
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                s = self._stage
            try:
                rss = self._total_rss()
            except psutil.NoSuchProcess:
                rss = 0
            self._samples.append((time.perf_counter() - self._t0, s, rss))
            self._stop.wait(self.interval)

    def start(self) -> None:
        global _active
        _active = self
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        global _active
        _active = None
        self.report()

    def report(self) -> None:
        by_stage: dict[str, list[int]] = {}
        for _, s, rss in self._samples:
            by_stage.setdefault(s, []).append(rss)
        print("=== memory by stage (RSS, tree-summed) ===")
        for s, vals in by_stage.items():
            print(
                f"  {s:14s} n={len(vals):5d}  avg={statistics.mean(vals) / 1e6:8.1f} MB  peak={max(vals) / 1e6:8.1f} MB"
            )
        peak = max(r for _, _, r in self._samples)
        print(f"  {'OVERALL':14s} peak={peak / 1e6:8.1f} MB")

import math
from collections.abc import Callable
from typing import Any, overload

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class SGD(Optimizer):
    def __init__(self, params: ParamsT, lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Negative learning rate: {lr}")
        defaults: dict[str, Any] = {"lr": lr}
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state: dict = self.state[p]
                t: int = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


if __name__ == "__main__":
    torch.manual_seed(57)
    initial_weights = 5 * torch.randn((10, 10))

    learning_rates = [1.0, 10.0, 100.0, 1000.0]
    num_steps = 10

    print("=" * 45)
    print(f"{'LR':<8} | {'Step':<6} | {'Loss':<15}")
    print("-" * 45)

    for lr in learning_rates:
        weights = torch.nn.Parameter(initial_weights.clone())
        opt = SGD([weights], lr=lr)
        for t in range(num_steps):
            opt.zero_grad()
            loss = (weights**2).mean()
            loss_val = loss.item()

            print(f"{lr:<8.1f} | {t:<6d} | {loss_val:<15.6f}")

            loss.backward()
            opt.step()
        print("-" * 45)

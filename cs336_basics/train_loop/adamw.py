import math
from collections.abc import Callable
from typing import Any, overload

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class AdamW(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float,
        weight_decay: float,
        eps: float = 1e-8,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> None:
        defaults: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay, "eps": eps, "betas": betas}
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        with torch.enable_grad():
            loss = None
            if closure is not None:
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            betas: tuple[float, float] = group["betas"]
            weight_decay: float = group["weight_decay"]
            eps: float = group["eps"]
            p: torch.Tensor
            for p in group["params"]:
                if p.grad is None:
                    continue

                # init
                state: dict = self.state[p]
                grad = p.grad
                t: int = state.get("t", 0)
                m: torch.Tensor = torch.zeros_like(p) if "m" not in state else state["m"]
                v: torch.Tensor = torch.zeros_like(p) if "v" not in state else state["v"]

                # update state
                state["m"] = m.lerp_(grad, 1 - betas[0])
                state["v"] = (v.mul_(betas[1])).addcmul_(grad, grad, value=(1 - betas[1]))
                state["t"] = t = t + 1

                # update params
                lr_adj = lr * math.sqrt(1 - pow(betas[1], t)) / (1 - pow(betas[0], t))
                p *= 1 - lr * weight_decay  # weight decay
                p.addcdiv_(m, torch.sqrt(v).add_(eps), value=-lr_adj)

        return loss

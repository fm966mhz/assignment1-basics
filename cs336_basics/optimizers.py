"""Optimizers."""

import math

from collections.abc import Callable
from typing import Any
from typing import Iterable
from typing import Optional
from typing import TypeAlias
from typing import Union

import torch

from torch import nn
from torch import optim

ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]


class SGD(optim.Optimizer):
    """Example SGD."""

    def __init__(self, params: ParamsT, lr: float = 1e-3):
        assert lr > 0, f"Invalid learning rate: {lr}."
        defaults = {"lr": lr}
        super().__init__(params=params, defaults=defaults)

    def step(  # type: ignore
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """Takes one step."""
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)  # The iteration number.
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(optim.Optimizer):
    """AdamW."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        assert lr > 0, f"Invalid learning rate: {lr}."
        assert weight_decay > 0, f"Invalid weight decay: {weight_decay}"
        assert eps > 0, f"Invalid eps: {eps}"
        assert betas[0] > 0 and betas[1] > 0, f"Invalid betas: {betas}"
        super().__init__(
            params=params,
            defaults={
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": betas,
                "eps": eps,
            },
        )

    def step(  # type: ignore
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            self._step_one_group(group)
        return loss

    def _step_one_group(self, param_group: dict[str, Any]):
        lr = param_group["lr"]
        weigtht_decay = param_group["weight_decay"]
        beta_1, beta_2 = param_group["betas"]
        eps = param_group["eps"]
        for p in param_group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            t = state.get("t", 1)
            m = state.get("m", 0)
            v = state.get("v", 0)
            grad = p.grad.data
            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * (grad**2)
            alpha_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)
            p.data -= alpha_t * m / torch.sqrt(v + eps)
            p.data *= 1 - lr * weigtht_decay
            state["t"] = t + 1
            state["m"] = m
            state["v"] = v

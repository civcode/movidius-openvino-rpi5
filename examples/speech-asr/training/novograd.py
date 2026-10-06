"""Minimal deterministic NovoGrad optimizer used by QuartzNet screens."""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class NovoGrad(Optimizer):
    """NovoGrad with per-tensor second-moment normalization.

    This follows the NVIDIA/NeMo-style formulation used for QuartzNet:
    normalize each parameter tensor's gradient by an EMA of its squared L2
    norm, apply coupled weight decay, then momentum.
    """

    def __init__(
        self,
        params,
        *,
        lr: float,
        betas: tuple[float, float] = (0.8, 0.5),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_averaging: bool = False,
    ):
        if lr <= 0:
            raise ValueError("NovoGrad lr must be > 0")
        if eps < 0:
            raise ValueError("NovoGrad eps must be >= 0")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("NovoGrad betas must be in [0,1)")
        if weight_decay < 0:
            raise ValueError("NovoGrad weight_decay must be >= 0")
        defaults = {
            "lr": float(lr),
            "betas": tuple(float(v) for v in betas),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "grad_averaging": bool(grad_averaging),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("NovoGrad does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros(
                        (),
                        device=parameter.device,
                        dtype=torch.float32,
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                norm = grad.detach().float().pow(2).sum()
                if state["step"] == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(norm, alpha=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                normalized = grad / denom.to(dtype=grad.dtype)
                if group["weight_decay"]:
                    normalized = normalized.add(
                        parameter,
                        alpha=group["weight_decay"],
                    )
                if group["grad_averaging"]:
                    normalized = normalized.mul(1.0 - beta1)

                exp_avg.mul_(beta1).add_(normalized)
                parameter.add_(exp_avg, alpha=-group["lr"])
                state["step"] += 1

        return loss

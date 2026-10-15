import math
from typing import Dict, List, Optional, Tuple
import torch
from torch.optim import AdamW, Optimizer


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    # Quintic Newton-Schulz iteration to orthogonalize 2D gradient matrices without SVD
    # Coefficients from Keller Jordan et al. (Muon optimizer)
    a, b, c = (3.4445, -4.7750,  2.0315)
    x = g.bfloat16() if g.dtype != torch.bfloat16 else g
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    # Normalize spectral norm estimate below 1.0
    norm = x.norm() + eps
    x = x / norm

    for _ in range(steps):
        a_x = a * x
        x_xt = x @ x.T
        b_x = b * (x_xt @ x)
        c_x = c * (x_xt @ (x_xt @ x))
        x = a_x + b_x + c_x

    if transposed:
        x = x.T
    return x.type_as(g)


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim != 2:
                    raise ValueError(f"Muon only supports 2D matrices, got shape {grad.shape}")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                update = grad.add(buf, alpha=momentum) if nesterov else buf
                ortho_update = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # Scale update by aspect ratio following Muon standard practice
                ortho_update *= max(1.0, math.sqrt(p.size(0) / p.size(1)))

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(ortho_update, alpha=-lr)

        return loss


def build_optimizers(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 0.002,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    optimizer_type: str = "muon_adamw",
) -> Tuple[Optional[Optimizer], Optimizer]:
    if optimizer_type == "adamw_only":
        decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2 and id(p) != id(model.tok_embeddings.weight)]
        no_decay_params = [p for p in model.parameters() if p.requires_grad and (p.ndim < 2 or id(p) == id(model.tok_embeddings.weight))]
        opt_adamw = AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=adamw_lr,
            betas=betas,
            eps=1e-8,
        )
        return None, opt_adamw

    # 2D hidden linear weights use Muon; embeddings, norms, and 1D params use AdamW
    muon_params = []
    adamw_decay_params = []
    adamw_no_decay_params = []

    # Exclude embeddings from Muon (even though they are 2D, token embedding coordinates are not matrix transforms)
    embedding_param_ids = {id(model.tok_embeddings.weight)}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in embedding_param_ids:
            adamw_no_decay_params.append(p)
        elif p.ndim == 2:
            muon_params.append(p)
        else:
            adamw_no_decay_params.append(p)

    if hasattr(torch.optim, "Muon"):
        muon_cls = torch.optim.Muon
    else:
        muon_cls = Muon

    opt_muon = muon_cls(muon_params, lr=muon_lr, weight_decay=weight_decay)
    adamw_groups = []
    if adamw_decay_params:
        adamw_groups.append({"params": adamw_decay_params, "weight_decay": weight_decay})
    if adamw_no_decay_params:
        adamw_groups.append({"params": adamw_no_decay_params, "weight_decay": 0.0})

    opt_adamw = AdamW(
        adamw_groups,
        lr=adamw_lr,
        betas=betas,
        eps=1e-8,
    )
    return opt_muon, opt_adamw


def get_lr_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
    stable_ratio: float = 0.7,
) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))

    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)

    if schedule == "cosine":
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay
    elif schedule == "wsd":
        # Warmup-Stable-Decay: keep peak LR until stable_ratio of total steps, then decay linearly
        if progress < stable_ratio:
            return 1.0
        decay_progress = (progress - stable_ratio) / (1.0 - stable_ratio)
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - decay_progress)
    else:
        return 1.0


def resume_rewarm_multiplier(
    sched_mult: float,
    step: int,
    resume_step: int,
    rewarm_steps: int,
    min_ratio: float = 0.1,
) -> float:
    # Blend LR from min_ratio back up to the schedule value over rewarm_steps
    # after a resume, so extending total_steps cannot spike LR on converged weights
    if rewarm_steps <= 0 or step < resume_step or step >= resume_step + rewarm_steps:
        return sched_mult
    ramp = float(step - resume_step + 1) / float(max(1, rewarm_steps))
    return min_ratio + (sched_mult - min_ratio) * ramp

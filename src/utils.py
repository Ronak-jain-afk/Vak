import os
import random
import subprocess
from typing import Any, Dict, Optional
import numpy as np
import torch


def get_gpu_temperature() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    opt_muon: torch.optim.Optimizer,
    opt_adamw: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    data_state: Dict[str, Any],
    val_loss: Optional[float] = None,
):
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    state = {
        "model": raw_model.state_dict(),
        "opt_muon": opt_muon.state_dict(),
        "opt_adamw": opt_adamw.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "data_state": data_state,
        "val_loss": val_loss,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    torch.save(state, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    opt_muon: Optional[torch.optim.Optimizer] = None,
    opt_adamw: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(state["model"])

    if opt_muon is not None and "opt_muon" in state:
        opt_muon.load_state_dict(state["opt_muon"])
    if opt_adamw is not None and "opt_adamw" in state:
        opt_adamw.load_state_dict(state["opt_adamw"])

    if "rng_torch" in state and state["rng_torch"] is not None:
        torch.set_rng_state(state["rng_torch"].cpu())
    if "rng_cuda" in state and state["rng_cuda"] is not None and torch.cuda.is_available():
        cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in state["rng_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    if "rng_numpy" in state:
        np.random.set_state(state["rng_numpy"])
    if "rng_python" in state:
        random.setstate(state["rng_python"])

    return state

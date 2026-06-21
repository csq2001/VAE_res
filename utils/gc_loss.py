import torch


def gc_balance_penalty(prob_gc: torch.Tensor, target: float = 0.5) -> torch.Tensor:
    return torch.mean((prob_gc - target) ** 2)

"""Optimizer factory for Phase 2 network training.

Phase 1 used hand-written optimizers (see ``workshoplib.optimizers``) so students
could read the update rules. Phase 2 trains real networks, so here we use the
fast, battle-tested ``torch.optim`` implementations - plus one custom optimizer,
MuON, which is not in PyTorch.

MuON ("MomentUm Orthogonalized by Newton-Schulz") takes the usual momentum
update and, for 2-D weight matrices, replaces it with the nearest orthogonal
matrix (computed cheaply with a few Newton-Schulz iterations) before stepping.
Intuitively this equalizes the update across all directions of a weight matrix
instead of letting a few large singular values dominate. It only makes sense for
matrix-shaped parameters, which is exactly why we deferred it from Phase 1 (where
each "parameter" was a single 2-D point) to Phase 2 (real Linear layers).
"""

from __future__ import annotations

import torch


def _newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize a 2-D matrix via a quintic Newton-Schulz iteration.

    Returns a matrix with (approximately) the same column/row space as ``matrix``
    but with singular values pushed toward 1. The quintic coefficients are the
    standard ones from Keller Jordan's Muon implementation.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix / (matrix.norm() + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        correction = b * gram + c * (gram @ gram)
        x = a * x + correction @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    """Momentum optimizer that orthogonalizes the update for 2-D weights.

    For each 2-D parameter (a Linear weight matrix) the momentum buffer is
    orthogonalized with Newton-Schulz before stepping, and scaled by
    ``sqrt(max(rows, cols))`` so the update size is comparable across layer
    shapes. For 1-D parameters (biases) there is nothing to orthogonalize, so we
    fall back to plain heavy-ball momentum SGD.

    This is a teaching-simplified version: real Muon also keeps embeddings and
    output heads on AdamW, but our small MLP has no such special layers.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.9, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
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
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if p.ndim >= 2:
                    update = _newton_schulz(buf, steps=ns_steps)
                    scale = max(p.shape[0], p.shape[1]) ** 0.5
                    p.add_(update, alpha=-lr * scale)
                else:
                    p.add_(buf, alpha=-lr)

        return loss


_DEFAULT_LR = {
    "sgd": 1.,
    "momentum": 0.5,
    "adagrad": 0.02,
    "adam": 1e-3,
    "muon": 0.01,
}


def make_optimizer(name: str, params, lr: float | None = None) -> torch.optim.Optimizer:
    """Create a torch optimizer by name (sgd/momentum/adagrad/adam/muon).

    Args:
        name: Optimizer key.
        params: Iterable of parameters to optimize (e.g. ``model.parameters()``).
        lr: Learning rate; defaults to a sensible per-optimizer value.
    """
    name = name.lower()
    if lr is None:
        if name not in _DEFAULT_LR:
            known = ", ".join(sorted(_DEFAULT_LR))
            raise ValueError(f"Unknown optimizer: {name!r}. Available: {known}.")
        lr = _DEFAULT_LR[name]

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "muon":
        return Muon(params, lr=lr)

    known = ", ".join(sorted(_DEFAULT_LR))
    raise ValueError(f"Unknown optimizer: {name!r}. Available: {known}.")

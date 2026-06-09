"""From-scratch optimizers for the Phase 1 low-dimensional demos.

These are deliberately simple re-implementations of common optimizers so that
students can read the exact update rule instead of trusting a black box. Each
optimizer keeps its own small state and updates a single parameter tensor in
place using that tensor's ``.grad``.

For Phase 2 (training real networks) we use the faster, battle-tested
``torch.optim`` versions via ``workshoplib.optimization`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from workshoplib.objectives import Objective


class SGD:
    """Plain gradient descent: ``p <- p - lr * grad``."""

    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, param: torch.Tensor) -> None:
        with torch.no_grad():
            param -= self.lr * param.grad


class Momentum:
    """SGD with heavy-ball momentum.

    ``v <- mu * v + grad`` then ``p <- p - lr * v``. The velocity ``v`` lets the
    step build up speed along consistent directions and damp oscillations.
    """

    def __init__(self, lr: float = 0.1, mu: float = 0.9):
        self.lr = lr
        self.mu = mu
        self.v: torch.Tensor | None = None

    def step(self, param: torch.Tensor) -> None:
        with torch.no_grad():
            if self.v is None:
                self.v = torch.zeros_like(param)
            self.v = self.mu * self.v + param.grad
            param -= self.lr * self.v


class AdaGrad:
    """Adaptive per-coordinate step sizes.

    Accumulates squared gradients ``G <- G + grad^2`` and scales the step by
    ``1 / sqrt(G + eps)``. Coordinates with large past gradients take smaller
    steps, which helps on ill-conditioned surfaces.
    """

    def __init__(self, lr: float = 0.5, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps
        self.g_sq: torch.Tensor | None = None

    def step(self, param: torch.Tensor) -> None:
        with torch.no_grad():
            if self.g_sq is None:
                self.g_sq = torch.zeros_like(param)
            self.g_sq = self.g_sq + param.grad**2
            param -= self.lr * param.grad / (torch.sqrt(self.g_sq) + self.eps)


class Adam:
    """Adam: momentum on the gradient plus AdaGrad-style scaling.

    Keeps an exponential moving average of the gradient (``m``) and of the
    squared gradient (``v``), with bias correction so early steps are not too
    small.
    """

    def __init__(
        self,
        lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.t = 0

    def step(self, param: torch.Tensor) -> None:
        with torch.no_grad():
            if self.m is None:
                self.m = torch.zeros_like(param)
                self.v = torch.zeros_like(param)
            self.t += 1
            grad = param.grad
            # momentum term : combination of current gradient and previous momentum : same dimension as the gradient 
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad 
            # AdaGrad term : combination of current gradient and previous gradient squared : same dimension as the gradient
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2 
            # Correction term : to account for the initial values of m and v being too small. m_hat and v_hat are approximately equal to m and v for large t.
            m_hat = self.m / (1.0 - self.beta1**self.t) # vector m is divided by a scalar
            v_hat = self.v / (1.0 - self.beta2**self.t) # vector v is divided by a scalar

            param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


# Default learning rates chosen so each optimizer behaves reasonably on the
# Phase 1 surfaces without per-objective tuning. The notebook overrides these
# when demonstrating learning-rate sensitivity.
_DEFAULT_LR = {
    "sgd": 0.05,
    "momentum": 0.02,
    "adagrad": 0.5,
    "adam": 0.1,
}


def make_optimizer(name: str, lr: float | None = None):
    """Create a from-scratch optimizer by name (sgd/momentum/adagrad/adam)."""
    name = name.lower()
    if lr is None:
        if name not in _DEFAULT_LR:
            known = ", ".join(sorted(_DEFAULT_LR))
            raise ValueError(f"Unknown optimizer: {name!r}. Available: {known}.")
        lr = _DEFAULT_LR[name]
    if name == "sgd":
        return SGD(lr=lr)
    if name == "momentum":
        return Momentum(lr=lr)
    if name == "adagrad":
        return AdaGrad(lr=lr)
    if name == "adam":
        return Adam(lr=lr)
    known = ", ".join(sorted(_DEFAULT_LR))
    raise ValueError(f"Unknown optimizer: {name!r}. Available: {known}.")


@dataclass
class Trajectory:
    """Result of running one optimizer on one objective.

    Args:
        name: The optimizer name (handy for plot legends).
        points: Tensor of shape ``(n_steps + 1, 2)`` with the visited points.
        losses: Tensor of shape ``(n_steps + 1,)`` with the loss at each point.
    """

    name: str
    points: torch.Tensor
    losses: torch.Tensor = field(repr=False)


def run_descent(
    objective: "Objective",
    optimizer_name: str,
    x0: tuple[float, float] | None = None,
    lr: float | None = None,
    n_steps: int = 50,
) -> Trajectory:
    """Run an optimizer on an objective and record its full trajectory.

    Args:
        objective: The 2-D surface to descend.
        optimizer_name: One of sgd/momentum/adagrad/adam.
        x0: Starting point; defaults to the objective's ``start``.
        lr: Learning rate. If omitted, the objective's recommended value is
            used, falling back to a global per-optimizer default.
        n_steps: Number of optimization steps to take.

    Returns:
        A ``Trajectory`` with the visited points and their losses.
    """
    start = objective.start if x0 is None else x0
    param = torch.tensor(start, dtype=torch.float32, requires_grad=True)

    if lr is None and objective.lrs is not None:
        lr = objective.lrs.get(optimizer_name.lower())
    optimizer = make_optimizer(optimizer_name, lr=lr)

    points = [param.detach().clone()]
    losses = [objective.value(param).detach().clone()]

    for _ in range(n_steps):
        if param.grad is not None:
            param.grad.zero_()
        loss = objective.value(param)
        loss.backward()
        optimizer.step(param)

        points.append(param.detach().clone())
        losses.append(objective.value(param).detach().clone())

    return Trajectory(
        name=optimizer_name,
        points=torch.stack(points),
        losses=torch.stack(losses),
    )

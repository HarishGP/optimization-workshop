"""Low-dimensional optimization objectives for Phase 1.

Each objective is a small 2-D test function used to visualize how different
optimizers move across a loss surface. Every objective exposes the same simple
interface so the trajectory runner and plotting helpers stay generic:

- ``value(xy)``: the loss at one or many points.
- ``bounds``: a sensible plotting window.
- ``minimizer``: the known optimum (or ``None`` for surfaces without one).
- ``start``: a default starting point that produces an interesting path.

The ``value`` functions are written to work elementwise, so they accept either
a single point of shape ``(2,)`` or a whole grid of shape ``(..., 2)``. This
lets us reuse the exact same function for taking gradient steps and for drawing
contour plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Objective:
    """A 2-D loss surface together with metadata for running and plotting.

    Args:
        name: Human-readable name shown in plots.
        value: Maps a tensor with last dimension 2 to a tensor of losses.
        bounds: ``((x_min, x_max), (y_min, y_max))`` plotting window.
        minimizer: Known optimum ``(x, y)``, or ``None`` if there is none.
        start: Default starting point ``(x, y)`` for trajectories.
        lrs: Recommended learning rate per optimizer for this surface. A single
            global learning rate cannot suit every surface (steep valleys need
            small steps, gentle bowls tolerate big ones), so each objective
            carries its own suggestions. ``None`` falls back to global defaults.
    """

    name: str
    value: Callable[[torch.Tensor], torch.Tensor]
    bounds: tuple[tuple[float, float], tuple[float, float]]
    minimizer: tuple[float, float] | None
    start: tuple[float, float]
    lrs: dict[str, float] | None = None


def _quadratic_bowl(xy: torch.Tensor) -> torch.Tensor:
    """Isotropic bowl ``f = x^2 + y^2``; the simplest possible surface."""
    x = xy[..., 0]
    y = xy[..., 1]
    return x**2 + y**2


def _ill_conditioned_quadratic(xy: torch.Tensor) -> torch.Tensor:
    """Stretched bowl ``f = x^2 + 25 y^2``.

    The two directions have very different curvature, so a single learning rate
    is either too large for the steep direction or too small for the shallow
    one. This is where momentum and adaptive methods start to pull ahead.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    return x**2 + 25.0 * y**2


def _rosenbrock(xy: torch.Tensor) -> torch.Tensor:
    """Classic banana valley ``f = (1 - x)^2 + 100 (y - x^2)^2``.

    The minimum at ``(1, 1)`` sits at the bottom of a long, curved valley that
    is easy to fall into but hard to follow.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


def _saddle(xy: torch.Tensor) -> torch.Tensor:
    """Saddle surface ``f = x^2 - y^2``.

    There is no minimum: the origin is a saddle point. Plain gradient descent
    can crawl very slowly near the flat ridge before escaping along ``y``.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    return x**2 - y**2


def _beale(xy: torch.Tensor) -> torch.Tensor:
    """Beale function; a harder multi-feature surface for keen students.

    Minimum at ``(3, 0.5)``. It has broad flat regions and sharp valleys.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    term1 = (1.5 - x + x * y) ** 2
    term2 = (2.25 - x + x * y**2) ** 2
    term3 = (2.625 - x + x * y**3) ** 2
    return term1 + term2 + term3


OBJECTIVES: dict[str, Objective] = {
    "quadratic_bowl": Objective(
        name="Quadratic bowl",
        value=_quadratic_bowl,
        bounds=((-2.5, 2.5), (-2.5, 2.5)),
        minimizer=(0.0, 0.0),
        start=(-2.0, 2.0),
        lrs={"sgd": 0.2, "momentum": 0.05, "adagrad": 0.8, "adam": 0.3},
    ),
    "ill_conditioned": Objective(
        name="Ill-conditioned quadratic",
        value=_ill_conditioned_quadratic,
        bounds=((-2.5, 2.5), (-1.0, 1.0)),
        minimizer=(0.0, 0.0),
        start=(-2.0, 0.8),
        lrs={"sgd": 0.02, "momentum": 0.01, "adagrad": 0.5, "adam": 0.2},
    ),
    "rosenbrock": Objective(
        name="Rosenbrock",
        value=_rosenbrock,
        bounds=((-2.0, 2.0), (-1.0, 3.0)),
        minimizer=(1.0, 1.0),
        start=(-1.5, 2.0),
        lrs={"sgd": 0.0008, "momentum": 0.0006, "adagrad": 0.2, "adam": 0.1},
    ),
    "saddle": Objective(
        name="Saddle",
        value=_saddle,
        bounds=((-2.0, 2.0), (-2.0, 2.0)),
        minimizer=None,
        start=(-1.5, 0.001),
        lrs={"sgd": 0.05, "momentum": 0.02, "adagrad": 0.3, "adam": 0.1},
    ),
    "beale": Objective(
        name="Beale",
        value=_beale,
        bounds=((-4.5, 4.5), (-4.5, 4.5)),
        minimizer=(3.0, 0.5),
        start=(0.0, -1.0),
        lrs={"sgd": 0.002, "momentum": 0.001, "adagrad": 0.1, "adam": 0.05},
    ),
}


def get_objective(name: str) -> Objective:
    """Look up an objective by key, with a helpful error on typos."""
    if name not in OBJECTIVES:
        known = ", ".join(sorted(OBJECTIVES))
        raise ValueError(f"Unknown objective: {name!r}. Available: {known}.")
    return OBJECTIVES[name]

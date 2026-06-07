"""Plotting helpers for the Phase 1 low-dimensional demos.

Two views are usually enough to build intuition:

- a contour map of the loss surface with each optimizer's path drawn on top, and
- a loss-versus-step curve comparing how fast each optimizer drives the loss down.

A simple 3-D surface plot is also provided to connect the contour view back to
the actual surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from workshoplib.objectives import Objective
    from workshoplib.optimizers import Trajectory


def _surface_grid(
    objective: "Objective", resolution: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the objective on a dense grid for plotting."""
    (x_min, x_max), (y_min, y_max) = objective.bounds
    xs = torch.linspace(x_min, x_max, resolution)
    ys = torch.linspace(y_min, y_max, resolution)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    grid_points = torch.stack([grid_x, grid_y], dim=-1)
    with torch.no_grad():
        grid_z = objective.value(grid_points)
    return grid_x.numpy(), grid_y.numpy(), grid_z.numpy()


def _contour_levels(grid_z: np.ndarray, num: int = 25) -> np.ndarray:
    """Pick contour levels, using log spacing for wide positive ranges."""
    z_min = float(grid_z.min())
    z_max = float(grid_z.max())
    if z_min > 0 and z_max / max(z_min, 1e-12) > 50:
        return np.logspace(np.log10(z_min), np.log10(z_max), num)
    return np.linspace(z_min, z_max, num)


def _as_list(trajectories: "Trajectory | Sequence[Trajectory]") -> list:
    """Allow callers to pass a single trajectory or a list of them."""
    if isinstance(trajectories, (list, tuple)):
        return list(trajectories)
    return [trajectories]


def plot_trajectory_on_contour(
    objective: "Objective",
    trajectories: "Trajectory | Sequence[Trajectory]",
    ax: plt.Axes | None = None,
    resolution: int = 200,
):
    """Draw the loss surface as filled contours with optimizer paths on top.

    Args:
        objective: The surface to draw.
        trajectories: One trajectory or several to overlay and compare.
        ax: Optional existing axis; a new figure is made if omitted.
        resolution: Grid resolution for the contour map.
    """
    trajectories = _as_list(trajectories)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    grid_x, grid_y, grid_z = _surface_grid(objective, resolution=resolution)
    levels = _contour_levels(grid_z)
    ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="viridis", alpha=0.7)
    ax.contour(grid_x, grid_y, grid_z, levels=levels, colors="white", linewidths=0.3)

    for traj in trajectories:
        pts = traj.points.numpy()
        ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=2.5, label=traj.name)
        ax.plot(pts[0, 0], pts[0, 1], marker="*", markersize=12, color="black")

    if objective.minimizer is not None:
        ax.plot(
            objective.minimizer[0],
            objective.minimizer[1],
            marker="X",
            markersize=11,
            color="red",
            label="minimum",
        )

    ax.set_title(objective.name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_loss_curves(
    trajectories: "Trajectory | Sequence[Trajectory]",
    ax: plt.Axes | None = None,
    log_scale: bool = True,
):
    """Plot loss versus step for one or more optimizers.

    Args:
        trajectories: One trajectory or several to compare.
        ax: Optional existing axis; a new figure is made if omitted.
        log_scale: Use a log y-axis when all losses are positive.
    """
    trajectories = _as_list(trajectories)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    all_positive = True
    for traj in trajectories:
        losses = traj.losses.numpy()
        ax.plot(losses, label=traj.name)
        if losses.min() <= 0:
            all_positive = False

    if log_scale and all_positive:
        ax.set_yscale("log")

    ax.set_title("Loss vs. step")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_surface_3d(objective: "Objective", resolution: int = 120, ax: plt.Axes | None = None):
    """Draw the objective as a 3-D surface to connect with the contour view."""
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")

    grid_x, grid_y, grid_z = _surface_grid(objective, resolution=resolution)
    ax.plot_surface(grid_x, grid_y, grid_z, cmap="viridis", alpha=0.9)
    ax.set_title(objective.name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("loss")
    return ax

"""First-layer alignment analysis for the Phase 2 MuON story.

The depth-``d`` ODT labels each point using ``2**d - 1`` oriented hyperplanes
(see ``workshoplib.odt``). A good first layer should devote neurons to *detecting*
those hyperplanes, i.e. each hidden neuron's input weight vector should line up
with one of the ODT hyperplane normals.

These helpers measure that alignment from the parameter snapshots recorded by
``workshoplib.training.train_model`` and visualize how different optimizers
allocate first-layer neurons across the ODT normals over training.

Key quantity: for first-layer weight matrix ``W`` (one row per neuron) and ODT
normals ``N`` (one row per hyperplane), the alignment matrix is
``A[i, j] = |cos(W_i, N_j)|`` - how well neuron ``i`` detects hyperplane ``j``
(absolute value because a hyperplane and its negation are the same boundary).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def neuron_hyperplane_alignment(weight, normals) -> np.ndarray:
    """Return ``|cos|`` between each neuron (row of ``weight``) and each normal.

    Args:
        weight: First-layer weights, shape ``(num_neurons, dim)`` (tensor or array).
        normals: ODT hyperplane normals, shape ``(num_hyperplanes, dim)``.

    Returns:
        Array of shape ``(num_neurons, num_hyperplanes)`` with values in ``[0, 1]``.
    """
    weight = np.asarray(weight, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    weight = weight / (np.linalg.norm(weight, axis=1, keepdims=True) + 1e-12)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    return np.abs(weight @ normals.T)


def _first_layer_key(params: dict) -> str:
    """Return the state-dict key of the first weight matrix (e.g. '0.weight')."""
    for key, tensor in params.items():
        if np.asarray(tensor).ndim == 2:
            return key
    raise ValueError("No 2-D weight matrix found in snapshot params.")


def _snapshot_at(history: dict, epoch: int) -> dict:
    """Return the snapshot recorded at ``epoch`` (raises if not captured)."""
    for snap in history["snapshots"]:
        if snap["epoch"] == epoch:
            return snap
    available = [s["epoch"] for s in history["snapshots"]]
    raise ValueError(f"No snapshot at epoch {epoch}. Available: {available}.")


def coverage_curve(history: dict, normals, threshold: float = 0.9) -> dict:
    """Track how many ODT hyperplanes are 'covered' by a neuron over training.

    A hyperplane is covered when at least one neuron aligns with its normal above
    ``threshold``. Uses the first-layer weights stored in each snapshot.

    Returns:
        Dict with ``epochs``, ``n_covered`` (count above threshold per epoch),
        and ``mean_best`` (mean over hyperplanes of the best neuron alignment).
    """
    epochs, n_covered, mean_best = [], [], []
    for snap in history["snapshots"]:
        weight = snap["params"][_first_layer_key(snap["params"])]
        alignment = neuron_hyperplane_alignment(weight, normals)
        best_per_hyperplane = alignment.max(axis=0)
        epochs.append(snap["epoch"])
        n_covered.append(int((best_per_hyperplane > threshold).sum()))
        mean_best.append(float(best_per_hyperplane.mean()))
    return {"epochs": epochs, "n_covered": n_covered, "mean_best": mean_best}


def plot_coverage(histories: dict, normals, threshold: float = 0.9):
    """Plot hyperplane coverage and mean best-alignment vs epoch, per optimizer.

    The left panel - how many of the ODT hyperplanes have a well-aligned neuron -
    is the clearest view of why one optimizer learns the target faster: it reaches
    full coverage in fewer epochs.
    """
    num_hyperplanes = np.asarray(normals).shape[0]
    fig, (ax_cov, ax_mean) = plt.subplots(1, 2, figsize=(13, 5))

    for name, history in histories.items():
        curve = coverage_curve(history, normals, threshold=threshold)
        ax_cov.plot(curve["epochs"], curve["n_covered"], marker="o", label=name)
        ax_mean.plot(curve["epochs"], curve["mean_best"], marker="o", label=name)

    ax_cov.axhline(num_hyperplanes, linestyle=":", color="gray")
    ax_cov.set_title(f"Hyperplanes covered (best |cos| > {threshold})")
    ax_cov.set_xlabel("epoch")
    ax_cov.set_ylabel(f"# covered (of {num_hyperplanes})")
    ax_cov.legend(fontsize=8)

    ax_mean.set_title("Mean best alignment over hyperplanes")
    ax_mean.set_xlabel("epoch")
    ax_mean.set_ylabel("mean over hyperplanes of max neuron |cos|")
    ax_mean.legend(fontsize=8)

    fig.tight_layout()
    return fig


def _sorted_alignment(weight, normals) -> np.ndarray:
    """Alignment matrix with neurons (rows) sorted by their top hyperplane."""
    alignment = neuron_hyperplane_alignment(weight, normals)
    order = sorted(
        range(alignment.shape[0]),
        key=lambda i: (int(alignment[i].argmax()), -float(alignment[i].max())),
    )
    return alignment[order]


def plot_alignment_heatmaps(histories: dict, normals, optimizers, epoch: int):
    """Heatmaps of neuron-to-hyperplane alignment at one epoch, per optimizer.

    Rows are neurons (sorted by their best-matching hyperplane), columns are the
    ODT hyperplanes with the root at column 0. Bright cells mean a neuron detects
    that hyperplane. A spread of bright cells across all columns means the layer
    covers the whole target; bright cells bunched in a few columns (with dark
    columns elsewhere) means neurons are crowding onto a few hyperplanes.
    """
    fig, axes = plt.subplots(
        1, len(optimizers), figsize=(6 * len(optimizers), 5), squeeze=False
    )
    for ax, name in zip(axes[0], optimizers):
        snap = _snapshot_at(histories[name], epoch)
        weight = snap["params"][_first_layer_key(snap["params"])]
        matrix = _sorted_alignment(weight, normals)
        image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{name} (epoch {epoch})")
        ax.set_xlabel("ODT hyperplane (0 = root)")
        ax.set_ylabel("neuron (sorted)")
        ax.set_xticks(range(matrix.shape[1]))
        fig.colorbar(image, ax=ax, fraction=0.046, label="|cos|")
    fig.tight_layout()
    return fig


def plot_assignment_counts(histories: dict, normals, optimizers, epoch: int, threshold: float = 0.5):
    """Bar chart of how many neurons are assigned to each ODT hyperplane.

    Each neuron is assigned to the hyperplane it aligns with most. Only neurons
    whose best alignment exceeds ``threshold`` are counted (others are not really
    detecting any hyperplane yet). This makes over-allocation (tall bars, often on
    the root) and gaps (zero-height bars = uncovered hyperplanes) visible.
    """
    num_hyperplanes = np.asarray(normals).shape[0]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(optimizers)

    for offset, name in enumerate(optimizers):
        snap = _snapshot_at(histories[name], epoch)
        weight = snap["params"][_first_layer_key(snap["params"])]
        alignment = neuron_hyperplane_alignment(weight, normals)
        assigned = alignment.argmax(axis=1)
        strong = alignment.max(axis=1) > threshold
        counts = np.bincount(assigned[strong], minlength=num_hyperplanes)
        positions = np.arange(num_hyperplanes) + offset * width
        ax.bar(positions, counts, width=width, label=name)

    ax.set_title(f"Neurons assigned per hyperplane (epoch {epoch}, |cos| > {threshold})")
    ax.set_xlabel("ODT hyperplane (0 = root)")
    ax.set_ylabel("# neurons")
    ax.set_xticks(np.arange(num_hyperplanes) + width * (len(optimizers) - 1) / 2)
    ax.set_xticklabels(range(num_hyperplanes))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

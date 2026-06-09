"""Mini-batch training loop and metrics for the Phase 2 ODT experiments.

A single ``train_model`` function trains a classifier with mini-batch gradient
descent and records per-epoch training/validation loss and accuracy, so the
notebook can compare optimizers by plotting their histories.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def _evaluate(model, x, y, loss_fn):
    """Return (loss, accuracy) of ``model`` on a dataset, without gradients."""
    model.eval()
    logits = model(x)
    loss = loss_fn(logits, y)
    accuracy = (logits.argmax(dim=1) == y).float().mean()
    model.train()
    return float(loss), float(accuracy)


@torch.no_grad()
def _snapshot_params(model):
    """Return a detached CPU copy of every model parameter, keyed by name."""
    return {name: p.detach().clone() for name, p in model.state_dict().items()}


def train_model(
    model,
    optimizer,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=40,
    batch_size=128,
    loss_fn=None,
    snapshot_every=None,
):
    """Train a classifier with mini-batch SGD-style updates and record metrics.

    Args:
        model: A classifier producing class logits.
        optimizer: Any optimizer from ``workshoplib.optimization.make_optimizer``.
        x_train, y_train: Training inputs (float) and labels (long).
        x_val, y_val: Validation inputs and labels.
        epochs: Number of passes over the training set.
        batch_size: Mini-batch size.
        loss_fn: Loss function; defaults to ``CrossEntropyLoss``.
        snapshot_every: If set, save a copy of all model parameters every this
            many epochs (plus the initial state at epoch 0 and the final state).
            Snapshots are stored in ``history["snapshots"]``.

    Returns:
        A ``history`` dict with lists ``train_loss``, ``val_loss``,
        ``train_acc``, ``val_acc`` (one entry per epoch). When ``snapshot_every``
        is set, it also contains ``snapshots``: a list of dicts, each with keys
        ``epoch`` (int) and ``params`` (a name -> tensor copy of the weights at
        that epoch).
    """
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if snapshot_every is not None:
        # Epoch 0 is the network's initial (untrained) state.
        history["snapshots"] = [{"epoch": 0, "params": _snapshot_params(model)}]

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = _evaluate(model, x_train, y_train, loss_fn)
        val_loss, val_acc = _evaluate(model, x_val, y_val, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if snapshot_every is not None and (
            epoch % snapshot_every == 0 or epoch == epochs
        ):
            history["snapshots"].append(
                {"epoch": epoch, "params": _snapshot_params(model)}
            )

    return history

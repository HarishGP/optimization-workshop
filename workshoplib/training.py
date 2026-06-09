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

    Returns:
        A ``history`` dict with lists ``train_loss``, ``val_loss``,
        ``train_acc``, ``val_acc`` (one entry per epoch).
    """
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for _ in range(epochs):
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

    return history

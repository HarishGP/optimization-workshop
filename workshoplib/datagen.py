import torch

from workshoplib.odt import generate_cob_odt_data


def make_toy_data(n=64):
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = 2 * x + 0.3 * torch.randn_like(x)
    return x, y


def make_odt_classification_data(
    num_data=10000,
    dim=50,
    depth=4,
    seed=0,
    val_fraction=0.2,
):
    """Generate an ODT classification dataset and split it into train/val.

    Inputs are sampled on the unit sphere and labelled by a single depth-``depth``
    oblique decision tree (see ``workshoplib.odt``). Because train and val share
    the same tree, the validation accuracy measures how well the network
    generalizes the *same* target function to unseen points.

    Labels from the ODT are ``{-1, +1}``; we map them to ``{0, 1}`` so they can be
    used directly with ``torch.nn.CrossEntropyLoss``.

    Args:
        num_data: Total number of points to generate before the split.
        dim: Input dimensionality.
        depth: ODT depth (depth 4 -> 16 leaves, 15 internal hyperplanes).
        seed: Random seed for reproducibility.
        val_fraction: Fraction of points held out for validation.

    Returns:
        ``(x_train, y_train, x_val, y_val, meta)`` where the ``x`` tensors are
        ``float32`` of shape ``(n, dim)`` and the ``y`` tensors are ``long`` of
        shape ``(n,)``. ``meta`` is the dict returned by the ODT generator.
    """
    x_np, y_np, _tree, meta = generate_cob_odt_data(
        num_data=num_data, dim=dim, depth=depth, seed=seed
    )

    x = torch.from_numpy(x_np).float()
    # Map ODT labels {-1, +1} -> class indices {0, 1}.
    y = torch.from_numpy(((y_np + 1) // 2).astype("int64"))

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=generator)
    x, y = x[perm], y[perm]

    n_val = int(round(val_fraction * x.shape[0]))
    x_val, y_val = x[:n_val], y[:n_val]
    x_train, y_train = x[n_val:], y[n_val:]

    return x_train, y_train, x_val, y_val, meta
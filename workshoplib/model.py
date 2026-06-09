import torch

def make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(1, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )


def make_mlp(input_dim=50, hidden_sizes=(16, 16), num_classes=2):
    """Build a fully connected ReLU classifier for the Phase 2 ODT task.

    Args:
        input_dim: Number of input features (ODT data dimensionality).
        hidden_sizes: Width of each hidden layer; its length sets the number of
            hidden layers (use 2 or 3 entries of 10-20 for the workshop).
        num_classes: Number of output classes (2 for the binary ODT labels).

    Returns:
        A ``torch.nn.Sequential`` ending in a linear layer of ``num_classes``
        logits, suitable for ``torch.nn.CrossEntropyLoss``.
    """
    layers = []
    in_features = input_dim
    for width in hidden_sizes:
        layers.append(torch.nn.Linear(in_features, width))
        layers.append(torch.nn.ReLU())
        in_features = width
    layers.append(torch.nn.Linear(in_features, num_classes))
    return torch.nn.Sequential(*layers)
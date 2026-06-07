import torch

def make_toy_data(n=64):
    x = torch.linspace(-1, 1, n).unsqueeze(1)
    y = 2 * x + 0.3 * torch.randn_like(x)
    return x, y
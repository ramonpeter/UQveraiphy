import math

import numpy as np
import torch


# Get some simple toy function
def generate_data(n_samples: int, unc: float = 0.2, seed: int | None = None):
    """
    Simple 1D regression problem based
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.concatenate(
        (
            np.random.uniform(-1, -0.22, n_samples // 2),
            np.random.uniform(0.22, 1, n_samples - n_samples // 2),
        )
    )
    y = 0.5 * np.sin(23 * x) + x / 2
    y_noise = y + np.random.randn(n_samples) * unc
    return x, y_noise


def kernel(x: torch.Tensor, y: torch.Tensor, sigma=None):
    "RBF kernel with median estimator"
    channels = len(x)
    dnorm2 = (
        (x.reshape(channels, 1, -1) - y.reshape(1, channels, -1)).square().sum(dim=2)
    )
    if sigma is None:
        sigma = torch.quantile(dnorm2.detach(), 0.5) / (2 * math.log(channels + 1))
    else:
        sigma = sigma
    return torch.exp(-dnorm2 / (2 * sigma))

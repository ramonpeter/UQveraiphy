import numpy as np
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


def brier_score(probs: Float[Array, "n"], labels: Int[Array, "n"]) -> Float[Array, ""]:
    """Brier score: mean squared error between predicted probability and label."""
    return jnp.mean((probs - labels) ** 2)


def expected_calibration_error(
    probs: Float[Array, "n"], labels: Int[Array, "n"], n_bins: int = 10
) -> float:
    """Expected Calibration Error: weighted average of |accuracy - confidence| per bin."""
    p = np.asarray(probs)
    y = np.asarray(labels)
    bin_edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))

    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p >= lo) & (p < hi)
        if lo == bin_edges[-2]:
            mask = mask | (p == hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y[mask].mean() - p[mask].mean())

    return ece / len(p)

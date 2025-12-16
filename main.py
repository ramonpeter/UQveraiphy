import math
from dataclasses import dataclass
from pprint import pprint
from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
from jaxtyping import Array, Float, PRNGKeyArray

from train import train_det_ens, train_det_net
from utils import (
    create_test_grid,
    create_uncertainty_plot,
    gen_data,
    plot_heatmap,
    pred_ensemble,
    uncertainty_decomposition,
)


def rbf_median(
    X: Float[Array, "n d"], Z: Float[Array, "m d"]
) -> Float[Array, "n m"]:
    """RBF kernel with median heuristic for bandwidth.

    K(x, z) = exp(-||x - z||² / (2σ²))
    where σ² = median(||x - z||²) / (2 log(n + 1))

    Args:
        X: First set of points [n, d]
        Z: Second set of points [m, d]

    Returns:
        Kernel matrix [n, m]
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Z, 2)
    chex.assert_equal(X.shape[1], Z.shape[1])

    # Compute pairwise squared distances
    dist2 = jax.vmap(
        jax.vmap(lambda x, z: jnp.sum(jnp.square(x - z)), in_axes=(0, None)),
        in_axes=(None, 0),
    )(X, Z)
    chex.assert_shape(dist2, (X.shape[0], Z.shape[0]))

    # Median heuristic for bandwidth
    sigma = jnp.median(dist2) / (2 * math.log(X.shape[0] + 1))

    kernel = jnp.exp(-dist2 / (2 * sigma))
    chex.assert_shape(kernel, (X.shape[0], Z.shape[0]))

    return kernel


@dataclass
class Config:
    """Configuration for uncertainty quantification experiments."""

    SEED: int = 12345
    save_name: str = "tmp"

    # Data parameters
    n_obs: int = 100
    noise: float = 0.1
    x_low_bound: float = -1.5
    x_up_bound: float = 2.5
    y_low_bound: float = -1.0
    y_up_bound: float = 1.5

    # Network parameters
    n_ens: int = 5
    width: int = 10
    depth: int = 3
    act: Literal["relu"] = "relu"

    # Training parameters
    n_steps: int = 1_000
    lr: float = 1e-3

    # Model selection
    model: Literal[
        "det_single",
        "det_ensemble",
        "det_repulsive",
        "bnn_single",
        "bnn_ensemble",
        "bnn_repulsive",
        "gp",
    ] = "det_ensemble"


def main(cfg: Config) -> None:
    """Main training and evaluation loop."""
    key = jrandom.PRNGKey(cfg.SEED)
    np.random.seed(cfg.SEED)

    # Generate data and grid
    data = gen_data(cfg.n_obs, cfg.noise)
    grid = create_test_grid(
        (cfg.x_low_bound, cfg.x_up_bound),
        (cfg.y_low_bound, cfg.y_up_bound),
        resolution=0.01,
    )
    pprint(cfg)

    optim = optax.adamw(cfg.lr)

    # Define network constructor
    if "det" in cfg.model:
        def get_det(key: PRNGKeyArray) -> eqx.nn.MLP:
            return eqx.nn.MLP(
                in_size=2,
                out_size=1,
                width_size=cfg.width,
                depth=cfg.depth,
                activation=jax.nn.relu,
                final_activation=jax.nn.sigmoid,
                key=key,
            )

    # Train and evaluate models
    if cfg.model == "det_single":
        key, det_key = jrandom.split(key)
        model = get_det(det_key)
        model = train_det_net(model, data, optim, cfg.n_steps)

        pred_grid = jax.vmap(model)(grid).squeeze()
        chex.assert_shape(pred_grid, (grid.shape[0],))

        plot_heatmap(pred_grid, grid, data, title="Deterministic Network")

    elif cfg.model == "det_ensemble":
        key, *ens_keys = jrandom.split(key, cfg.n_ens + 1)
        model = eqx.filter_vmap(get_det)(jnp.array(ens_keys))
        model = train_det_ens(model, data, optim, cfg.n_steps)

        # Predictions: [n_grid, n_ensemble]
        pred_grid = jax.vmap(lambda x: pred_ensemble(model, x))(grid).squeeze()
        chex.assert_rank(pred_grid, 2)
        chex.assert_equal(pred_grid.shape[0], grid.shape[0])

        # Uncertainty decomposition
        total, aleatoric, epistemic = uncertainty_decomposition(pred_grid)

        aspect_ratio = (cfg.x_up_bound - cfg.x_low_bound) / (
            cfg.y_up_bound - cfg.y_low_bound
        )

        create_uncertainty_plot(
            total,
            aleatoric,
            epistemic,
            grid,
            data,
            aspect_ratio,
            shared_colorbar=True,
            title_prefix="Det Ensemble",
        )

    elif cfg.model == "det_repulsive":
        raise NotImplementedError("det_repulsive not yet implemented")
    elif cfg.model == "bnn_single":
        raise NotImplementedError("bnn_single not yet implemented")
    elif cfg.model == "bnn_repulsive":
        raise NotImplementedError("bnn_repulsive not yet implemented")
    elif cfg.model == "bnn_ensemble":
        raise NotImplementedError("bnn_ensemble not yet implemented")
    elif cfg.model == "gp":
        raise NotImplementedError("gp not yet implemented")
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    plt.tight_layout()
    plt.savefig(f"{cfg.save_name}.pdf", dpi=150, bbox_inches="tight")
    print(f"Saved plot to {cfg.save_name}.pdf")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)

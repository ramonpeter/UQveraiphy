import math
from dataclasses import dataclass
from pprint import pprint
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
from jaxtyping import Array

from train import train_det_net, train_det_ens
from utils import (
    gen_data,
    create_test_grid,
    plot_heatmap,
    pred_ensemble,
    uncertainty_decomposition,
)


def rbf_median(X:Array,Z:Array) -> Array:
    assert X.shape[1] == Z.shape[1]
    dist2 = jax.vmap(
        jax.vmap(lambda x, z: jnp.sum(jnp.square(x - z)), in_axes=(0, None)),
        in_axes=(None, 0),
    )(X, Z)
    sigma = jnp.median(dist2)/(2 * math.log(X.shape[0] + 1))
    return jnp.exp(-dist2 / (2*sigma))



# Config
@dataclass
class Config:
    SEED: int = 12345

    # data
    n_obs: int = 100
    noise: float = 0.1
    x_low_bound: float = -1.5
    x_up_bound: float = 2.5
    y_low_bound: float = -1.0
    y_up_bound: float = 1.5

    # nets
    n_ens: int = 5
    width: int = 10
    depth: int = 3
    act: Literal["relu"] = "relu"

    # training
    n_steps: int = 1_000
    lr: float = 1e-3

    model: Literal[
        "det_single",
        "det_ensemble",
        "det_repulsive",
        "bnn_single",
        "bnn_ensemble",
        "bnn_repulsive",
        "gp",
    ] = "det_ensemble"


def main(cfg: Config):
    key = jrandom.PRNGKey(cfg.SEED)
    np.random.seed(cfg.SEED)

    data = gen_data(cfg.n_obs, cfg.noise)
    grid = create_test_grid(
        (cfg.x_low_bound, cfg.x_up_bound),
        (cfg.y_low_bound, cfg.y_up_bound),
        resolution=0.01,
    )
    pprint(cfg)
    optim = optax.adamw(cfg.lr)

    if "det" in cfg.model:
        get_det = lambda key: eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=cfg.width,
            depth=cfg.depth,
            activation=jax.nn.relu,
            final_activation=jax.nn.sigmoid,
            key=key,
        )

    if cfg.model == "det_single":
        key, det_key = jrandom.split(key)
        model = get_det(det_key)
        model = train_det_net(model, data, optim, cfg.n_steps)
        pred_grid = jax.vmap(model)(grid)
        plot_heatmap(pred_grid, grid, data, title="Deterministic net")
    elif cfg.model == "det_ensemble":
        key, *ens_keys = jrandom.split(key, cfg.n_ens + 1)
        model = eqx.filter_vmap(get_det)(jnp.array(ens_keys))
        model = train_det_ens(model, data, optim, cfg.n_steps)

        pred_grid = jax.vmap(lambda x: pred_ensemble(model, x))(grid).squeeze()
        total, aleatoric, epistemic = uncertainty_decomposition(pred_grid)

        aspect_ratio = (cfg.x_up_bound - cfg.x_low_bound) / (
            cfg.y_up_bound - cfg.y_low_bound
        )
        # fig, axes = plt.subplots(1,3, figsize=(5*aspect_ratio * 3, 5))
        # plt.subplot(131)
        # plot_heatmap(total, grid, data, title="Det Ensemble: total")
        # plt.subplot(132)
        # plot_heatmap(aleatoric, grid, data, title="Det Ensemble: aleatoric")
        # plt.subplot(133)
        # plot_heatmap(epistemic, grid, data, title="Det Ensemble: epistemic")

        # TODO: Add the predictive as well

        shared_colorbar = True

        if shared_colorbar:
            # Find global min/max across all plots
            vmin = min(total.min(), aleatoric.min(), epistemic.min())
            vmax = max(total.max(), aleatoric.max(), epistemic.max())

            # Adjust figure width to account for single colorbar
            fig, axes = plt.subplots(1, 3, figsize=(5 * aspect_ratio * 3 + 1, 5))

            plt.subplot(131)
            plot_heatmap(
                total, grid, data, title="Det Ensemble: total", vmin=vmin, vmax=vmax
            )
            plt.subplot(132)
            plot_heatmap(
                aleatoric,
                grid,
                data,
                title="Det Ensemble: aleatoric",
                vmin=vmin,
                vmax=vmax,
            )
            plt.subplot(133)
            sc = plot_heatmap(
                epistemic,
                grid,
                data,
                title="Det Ensemble: epistemic",
                vmin=vmin,
                vmax=vmax,
            )

            # Add single colorbar on the right
            # fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.04)

            # Add single colorbar on the right
            plt.tight_layout()
            fig.subplots_adjust(right=1.1)
            cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
            fig.colorbar(sc, cax=cbar_ax)

        else:
            # Individual colorbars for each subplot
            fig, axes = plt.subplots(1, 3, figsize=(5 * aspect_ratio * 3 + 1.5, 5))

            plt.subplot(131)
            plot_heatmap(
                total, grid, data, title="Det Ensemble: total", show_colorbar=True
            )
            plt.subplot(132)
            plot_heatmap(
                aleatoric,
                grid,
                data,
                title="Det Ensemble: aleatoric",
                show_colorbar=True,
            )
            plt.subplot(133)
            plot_heatmap(
                epistemic,
                grid,
                data,
                title="Det Ensemble: epistemic",
                show_colorbar=True,
            )

        plt.tight_layout()

    elif cfg.model == "det_repulsive":
        pass
    elif cfg.model == "bnn_single":
        pass
    elif cfg.model == "bnn_repulsive":
        pass
    elif cfg.model == "bnn_ensemble":
        pass
    elif cfg.model == "gp":
        pass
    else:
        
        raise NotImplementedError(f"{cfg.model} is not implemented.")

    plt.tight_layout()
    plt.savefig("tmp.png")


if __name__ == "__main__":
    tyro.cli(main)

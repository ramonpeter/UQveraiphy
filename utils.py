import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
import matplotlib.pyplot as plt
import sklearn


# Helper functions
cat_entropy = lambda p: -jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0))
ber_entropy = lambda p: -jnp.where(
    (p > 0) & (p < 1), p * jnp.log(p) + (1 - p) * jnp.log(1 - p), 0.0
)
bin_ce = lambda p, label: -jnp.where(
    label, jnp.log(p), jnp.log(1 - p)
)  # could also be turned into a sum


@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def pred_ensemble(model, x):
    return model(x)


def uncertainty_decomposition(prob: Array) -> tuple[Array, Array, Array]:
    assert prob.ndim > 1
    entr_exp = ber_entropy(prob.mean(1))  # total
    exp_entr = jax.vmap(ber_entropy, in_axes=1, out_axes=1)(prob).mean(1)  # alea
    diff = entr_exp - exp_entr  # epi
    return entr_exp, exp_entr, diff


# Data functions
def gen_data(n_obs: int, noise: float):
    X, label = sklearn.datasets.make_moons(n_samples=n_obs, noise=noise)
    return {"X": X, "label": label}


def create_test_grid(x_range, y_range, resolution=0.1):
    x = jnp.arange(x_range[0], x_range[1] + resolution / 2, resolution)
    y = jnp.arange(y_range[0], y_range[1] + resolution / 2, resolution)
    X, Y = jnp.meshgrid(x, y)
    return jnp.stack([X.ravel(), Y.ravel()], axis=1)


# Plotting functions
def plot_data(data):
    X, label = data["X"], data["label"]
    plt.scatter(X[label == 0, 0], X[label == 0, 1])
    plt.scatter(X[label == 1, 0], X[label == 1, 1])
    plt.axis("off")
    plt.show()


# def plot_heatmap(pred, grid, data, title: str = ""):
#     plt.scatter(grid[:, 0], grid[:, 1], c=pred, cmap="viridis")
#     plt.xlim(grid[:, 0].min(), grid[:, 0].max())
#     plt.ylim(grid[:, 1].min(), grid[:, 1].max())
#     if title:
#         plt.title(title)
#     plot_data(data)
#     plt.tight_layout()


def plot_heatmap(
    pred, grid, data, title: str = "", vmin=None, vmax=None, show_colorbar=False
):
    sc = plt.scatter(
        grid[:, 0], grid[:, 1], c=pred, cmap="viridis", vmin=vmin, vmax=vmax
    )
    plt.xlim(grid[:, 0].min(), grid[:, 0].max())
    plt.ylim(grid[:, 1].min(), grid[:, 1].max())
    plt.gca().set_aspect("equal", adjustable="box")
    if title:
        plt.title(title)
    plot_data(data)
    if show_colorbar:
        plt.colorbar(sc)
    return sc

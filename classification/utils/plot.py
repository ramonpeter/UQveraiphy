import jax
from jax import numpy as jnp
from jaxtyping import Float, Array
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .utils import ber_entropy, uncertainty_decomposition


def _create_class_colormap():
    colors = plt.cm.tab10.colors[:2]
    return LinearSegmentedColormap.from_list("class_colors", [colors[0], colors[1]])


def _plot_subplot(ax, grid, vals, X, y, cmap, vmin, vmax, x_lim, y_lim, is_first_plot):
    ax.scatter(
        grid[:, 0],
        grid[:, 1],
        c=vals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    alpha = 1.0 if is_first_plot else 0.7
    linewidth = 0.8 if is_first_plot else 0.4

    for c in range(2):
        ax.scatter(
            X[y == c, 0],
            X[y == c, 1],
            s=20,
            alpha=alpha,
            edgecolors="black",
            linewidths=linewidth,
            rasterized=False,
        )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def plot_res_det(
    data: dict,
    grid: Float[Array, "n 2"],
    logits: Float[Array, "n"],
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    title: str | None = None,
):
    preds = jax.nn.sigmoid(logits)
    pred_mean = preds
    pred_entr = ber_entropy(preds)

    X, y = data["X"], data["label"]
    cmap = _create_class_colormap()

    fig = plt.figure(figsize=(8, 4))
    if title:
        fig.suptitle(title, y=0.9)
    plt.subplots_adjust(wspace=0.05, hspace=0)

    max_entropy = -jnp.log(0.5)

    for i, (vals, current_cmap, vmin, vmax) in enumerate(
        [(pred_mean, cmap, 0, 1), (pred_entr, "viridis", 0, max_entropy)], 1
    ):
        ax = plt.subplot(1, 2, i)
        _plot_subplot(
            ax, grid, vals, X, y, current_cmap, vmin, vmax, x_lim, y_lim, i == 1
        )


def plot_res_sample(
    data: dict,
    grid: Float[Array, "n 2"],
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    logits: Float[Array, "n samples"] | None = None,
    decomposed: dict[str, Float[Array, "n"]] | None = None,
    title: str | None = None,
):
    if decomposed:
        pred_mean = decomposed["probs"]
        pred_entr = decomposed["total"]
        aleatoric = decomposed["aleatoric"]
        epistemic = decomposed["epistemic"]
    else:
        assert logits is not None, "If decomposed is None, logits need to be provided"
        preds = jax.nn.sigmoid(logits)
        pred_mean = preds.mean(1)
        pred_entr, aleatoric, epistemic = uncertainty_decomposition(preds)

    X, y = data["X"], data["label"]
    cmap = _create_class_colormap()

    fig = plt.figure(figsize=(16, 4))
    if title:
        fig.suptitle(title, y=0.9)
    plt.subplots_adjust(wspace=0.05, hspace=0)

    max_entropy = -jnp.log(0.5)

    for i, (vals, current_cmap, vmin, vmax) in enumerate(
        [
            (pred_mean, cmap, 0, 1),
            (pred_entr, "viridis", 0, max_entropy),
            (aleatoric, "viridis", 0, max_entropy),
            (epistemic, "viridis", 0, max_entropy),
        ],
        1,
    ):
        ax = plt.subplot(1, 4, i)
        _plot_subplot(
            ax, grid, vals, X, y, current_cmap, vmin, vmax, x_lim, y_lim, i == 1
        )

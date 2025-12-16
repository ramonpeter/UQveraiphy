import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import sklearn.datasets
from jaxtyping import Array, Float, Int, PRNGKeyArray
from matplotlib.collections import PathCollection

from bnn import BNN


# ============================================================================
# Helper functions
# ============================================================================

def cat_entropy(p: Float[Array, "..."]) -> Float[Array, "..."]:
    """Categorical entropy: H(p) = -∑ p log(p)"""
    return -jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0))


def ber_entropy(p: Float[Array, "..."]) -> Float[Array, "..."]:
    """Bernoulli entropy: H(p) = -p log(p) - (1-p) log(1-p)"""
    return -jnp.where(
        (p > 0) & (p < 1), p * jnp.log(p) + (1 - p) * jnp.log(1 - p), 0.0
    )


def bin_ce(p: Float[Array, "..."], label: Int[Array, "..."]) -> Float[Array, "..."]:
    """Binary cross-entropy: -[y log(p) + (1-y) log(1-p)]"""
    chex.assert_equal_shape([p, label])
    return -jnp.where(label, jnp.log(p), jnp.log(1 - p))


@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def pred_ensemble(model: eqx.nn.MLP, x: Float[Array, "d"]) -> Float[Array, ""]:
    """Predict with ensemble member."""
    return model(x)


def pred_bnn_samples(
    model: BNN,  # BNN type
    x: Float[Array, "d"],
    n_samples: int,
    key: PRNGKeyArray,
) -> Float[Array, "n_samples"]:
    """Get multiple predictions from BNN for uncertainty estimation.

    Args:
        model: BNN network
        x: Input point [d]
        n_samples: Number of samples to draw
        key: PRNG key

    Returns:
        Predictions [n_samples]
    """
    keys = jrandom.split(key, n_samples)
    return jax.vmap(lambda k: model(x, k))(keys).squeeze()


def pred_bnn_ensemble(
    model,  # BNN type
    x: Float[Array, "n d"],
    n_samples: int,
    key: PRNGKeyArray,
) -> Float[Array, "n n_samples"]:
    """Get ensemble predictions for multiple inputs.

    Args:
        model: BNN network
        x: Input points [n, d]
        n_samples: Number of samples per input
        key: PRNG key

    Returns:
        Predictions [n, n_samples]
    """
    keys = jrandom.split(key, x.shape[0])
    return jax.vmap(lambda xi, ki: pred_bnn_samples(model, xi, n_samples, ki))(x, keys)


def uncertainty_decomposition_bnn(
    prob: Float[Array, "n_samples n_mc_samples"]
) -> tuple[Float[Array, "n_samples"], Float[Array, "n_samples"], Float[Array, "n_samples"]]:
    """Decompose BNN predictive uncertainty (same as ensemble version).

    Args:
        prob: Predictions from BNN samples [n_samples, n_mc_samples]

    Returns:
        total: Total uncertainty (entropy of mean prediction)
        aleatoric: Aleatoric uncertainty (expected entropy)
        epistemic: Epistemic uncertainty (difference)
    """
    return uncertainty_decomposition(prob)


def uncertainty_decomposition(
    prob: Float[Array, "n_samples n_ensemble"]
) -> tuple[Float[Array, "n_samples"], Float[Array, "n_samples"], Float[Array, "n_samples"]]:
    """Decompose predictive uncertainty into aleatoric and epistemic components.

    Total uncertainty = Aleatoric + Epistemic
    H[E[p]] = E[H[p]] + (H[E[p]] - E[H[p]])

    Args:
        prob: Predictions from ensemble [n_samples, n_ensemble]

    Returns:
        total: Total uncertainty (entropy of mean prediction)
        aleatoric: Aleatoric uncertainty (expected entropy)
        epistemic: Epistemic uncertainty (difference)
    """
    chex.assert_rank(prob, 2)

    # Total uncertainty: entropy of expected prediction
    entr_exp = ber_entropy(prob.mean(axis=1))
    chex.assert_shape(entr_exp, (prob.shape[0],))

    # Aleatoric uncertainty: expected entropy
    exp_entr = jax.vmap(ber_entropy, in_axes=1, out_axes=1)(prob).mean(axis=1)
    chex.assert_shape(exp_entr, (prob.shape[0],))

    # Epistemic uncertainty: mutual information
    epistemic = entr_exp - exp_entr
    chex.assert_shape(epistemic, (prob.shape[0],))

    return entr_exp, exp_entr, epistemic


# ============================================================================
# Data functions
# ============================================================================

def gen_data(
    n_obs: int, noise: float
) -> dict[str, Float[Array, "n_obs ..."] | Int[Array, "n_obs"]]:
    """Generate two moons dataset.

    Args:
        n_obs: Number of observations
        noise: Noise level

    Returns:
        Dictionary with 'X' [n_obs, 2] and 'label' [n_obs]
    """
    X, label = sklearn.datasets.make_moons(n_samples=n_obs, noise=noise)
    X = jnp.array(X)
    label = jnp.array(label, dtype=jnp.int32)

    chex.assert_shape(X, (n_obs, 2))
    chex.assert_shape(label, (n_obs,))

    return {"X": X, "label": label}


def create_test_grid(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float = 0.1,
) -> Float[Array, "n_points 2"]:
    """Create a 2D grid for visualization.

    Args:
        x_range: (min, max) for x-axis
        y_range: (min, max) for y-axis
        resolution: Grid spacing

    Returns:
        Grid points [n_points, 2]
    """
    x = jnp.arange(x_range[0], x_range[1] + resolution / 2, resolution)
    y = jnp.arange(y_range[0], y_range[1] + resolution / 2, resolution)
    X, Y = jnp.meshgrid(x, y)
    grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    chex.assert_rank(grid, 2)
    chex.assert_equal(grid.shape[1], 2)

    return grid


# ============================================================================
# Plotting functions
# ============================================================================

def plot_data(
    data: dict[str, Float[Array, "n ..."] | Int[Array, "n"]],
    alpha: float = 0.6,
    s: float = 40,
) -> None:
    """Plot binary classification data.

    Args:
        data: Dictionary with 'X' [n, 2] and 'label' [n]
        alpha: Point transparency
        s: Point size
    """
    X, label = data["X"], data["label"]
    chex.assert_rank(X, 2)
    chex.assert_rank(label, 1)
    chex.assert_equal(X.shape[1], 2)
    chex.assert_equal(X.shape[0], label.shape[0])

    plt.scatter(X[label == 0, 0], X[label == 0, 1], alpha=alpha, s=s, marker="x", label="Class 0", c="red")
    plt.scatter(X[label == 1, 0], X[label == 1, 1], alpha=alpha, s=s, marker="o", label="Class 1", c="orange")
    plt.axis("off")


def plot_heatmap(
    pred: Float[Array, "n"],
    grid: Float[Array, "n 2"],
    data: dict[str, Float[Array, "m ..."] | Int[Array, "m"]],
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = False,
    cmap: str = "viridis",
    s: float = 10,
) -> PathCollection:
    """Plot heatmap with data overlay.

    Args:
        pred: Predictions for grid points [n]
        grid: Grid coordinates [n, 2]
        data: Training data dictionary
        title: Plot title
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        show_colorbar: Whether to show colorbar
        cmap: Colormap name
        s: Scatter point size

    Returns:
        PathCollection for colorbar creation
    """
    chex.assert_rank(pred, 1)
    chex.assert_rank(grid, 2)
    chex.assert_equal(pred.shape[0], grid.shape[0])
    chex.assert_equal(grid.shape[1], 2)

    sc = plt.scatter(
        grid[:, 0], grid[:, 1], c=pred, cmap=cmap, vmin=vmin, vmax=vmax, s=s
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


def create_uncertainty_plot(
    total: Float[Array, "n"],
    aleatoric: Float[Array, "n"],
    epistemic: Float[Array, "n"],
    grid: Float[Array, "n 2"],
    data: dict[str, Float[Array, "m ..."] | Int[Array, "m"]],
    aspect_ratio: float,
    shared_colorbar: bool = True,
    title_prefix: str = "Det Ensemble",
) -> None:
    """Create 3-panel uncertainty decomposition plot.

    Args:
        total: Total uncertainty [n]
        aleatoric: Aleatoric uncertainty [n]
        epistemic: Epistemic uncertainty [n]
        grid: Grid coordinates [n, 2]
        data: Training data
        aspect_ratio: Width/height ratio
        shared_colorbar: Use shared colorbar across subplots
        title_prefix: Prefix for subplot titles
    """
    chex.assert_equal_shape([total, aleatoric, epistemic])
    chex.assert_rank(grid, 2)
    chex.assert_equal(total.shape[0], grid.shape[0])

    if shared_colorbar:
        vmin = min(total.min(), aleatoric.min(), epistemic.min())
        vmax = max(total.max(), aleatoric.max(), epistemic.max())

        fig, axes = plt.subplots(1, 3, figsize=(5 * aspect_ratio * 3 + 1, 5))

        plt.subplot(131)
        plot_heatmap(total, grid, data, title=f"{title_prefix}: Total", vmin=vmin, vmax=vmax)

        plt.subplot(132)
        plot_heatmap(aleatoric, grid, data, title=f"{title_prefix}: Aleatoric", vmin=vmin, vmax=vmax)

        plt.subplot(133)
        sc = plot_heatmap(epistemic, grid, data, title=f"{title_prefix}: Epistemic", vmin=vmin, vmax=vmax)

        plt.tight_layout()
        fig.subplots_adjust(right=1.3)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
        fig.colorbar(sc, cax=cbar_ax)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(5 * aspect_ratio * 3 + 1.5, 5))

        plt.subplot(131)
        plot_heatmap(total, grid, data, title=f"{title_prefix}: Total", show_colorbar=True)

        plt.subplot(132)
        plot_heatmap(aleatoric, grid, data, title=f"{title_prefix}: Aleatoric", show_colorbar=True)

        plt.subplot(133)
        plot_heatmap(epistemic, grid, data, title=f"{title_prefix}: Epistemic", show_colorbar=True)

        plt.tight_layout()

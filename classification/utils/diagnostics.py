import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from jaxtyping import Float, Int, Array


def brier_score(
    probs: Float[Array, "n"], labels: Int[Array, "n"]
) -> Float[Array, ""]:
    """Brier score: mean squared error between predicted probability and label."""
    return jnp.mean((probs - labels) ** 2)


def reliability_curve(
    probs: Float[Array, "n"], labels: Int[Array, "n"], n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve (predicted prob vs observed frequency).

    Returns (bin_centers, bin_accuracies, bin_counts) as numpy arrays.
    Empty bins are excluded.
    """
    p = np.asarray(probs)
    y = np.asarray(labels)
    bin_edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))

    centers, accs, counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p >= lo) & (p < hi)
        if lo == bin_edges[-2]:  # include right edge in last bin
            mask = mask | (p == hi)
        if mask.sum() == 0:
            continue
        centers.append(p[mask].mean())
        accs.append(y[mask].mean())
        counts.append(mask.sum())

    return np.array(centers), np.array(accs), np.array(counts)


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


def conformal_prediction_sets(
    cal_probs: Float[Array, "n_cal"],
    cal_labels: Int[Array, "n_cal"],
    test_probs: Float[Array, "n_test"],
    alpha: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Split conformal prediction for binary classification.

    Nonconformity score: 1 - predicted probability of the true class.

    Returns (set_sizes, q_hat) where set_sizes is an int array with values in {0, 1, 2}.
    """
    cal_p = np.asarray(cal_probs)
    cal_y = np.asarray(cal_labels)
    test_p = np.asarray(test_probs)

    # Nonconformity scores on calibration set
    scores = np.where(cal_y == 1, 1 - cal_p, cal_p)

    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(scores, np.clip(q_level, 0, 1), method="higher")

    # For each test point, check which classes are in the prediction set
    # Class 1 in set if score for class 1 <= q_hat, i.e. 1 - p <= q_hat
    # Class 0 in set if score for class 0 <= q_hat, i.e. p <= q_hat
    class1_in = (1 - test_p) <= q_hat
    class0_in = test_p <= q_hat
    set_sizes = class1_in.astype(int) + class0_in.astype(int)

    return set_sizes, float(q_hat)


def plot_reliability_diagram(
    methods_dict: dict[str, Float[Array, "n"]],
    y_true: Int[Array, "n"],
    n_bins: int = 10,
    save_path: str | None = None,
):
    """Reliability diagram with all methods overlaid.

    Args:
        methods_dict: {method_name: predicted_probabilities}
        y_true: true binary labels
        n_bins: number of calibration bins
        save_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.08, 0.08, 0.98, 0.98))

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")

    markers = ["o", "s", "D", "^", "v"]
    for i, (name, probs) in enumerate(methods_dict.items()):
        centers, accs, counts = reliability_curve(probs, y_true, n_bins)
        bs = float(brier_score(probs, y_true))
        ece = expected_calibration_error(probs, y_true, n_bins)
        label = f"{name} (BS={bs:.3f}, ECE={ece:.3f})"
        ax.plot(
            centers,
            accs,
            marker=markers[i % len(markers)],
            label=label,
            markersize=6,
            linewidth=1.5,
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_conformal_sets(
    grid: Float[Array, "n 2"],
    set_sizes: np.ndarray,
    grid_probs: Float[Array, "n"],
    data: dict,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    title: str | None = None,
    ax: plt.Axes | None = None,
):
    """Heatmap of conformal prediction set sizes on a 2D grid.

    Set size 1: colored by predicted class (blue/orange).
    Set size 2: purple (ambiguous).
    Set size 0: should not occur at nominal level.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    grid_np = np.asarray(grid)
    sizes = np.asarray(set_sizes)
    probs = np.asarray(grid_probs)

    # Build color array: size 1 → predicted class color, size 2 → purple
    tab10 = plt.cm.tab10.colors
    color_class0 = np.array(tab10[0])  # blue
    color_class1 = np.array(tab10[1])  # orange
    color_ambiguous = np.array([0.7, 0.5, 0.85])  # light purple

    rgba = np.zeros((len(sizes), 4))
    rgba[:, 3] = 1.0  # alpha

    mask1 = sizes == 1
    mask2 = sizes == 2

    # For set size 1, color by predicted class
    pred_class = (probs > 0.5).astype(int)
    rgba[mask1 & (pred_class == 0), :3] = color_class0
    rgba[mask1 & (pred_class == 1), :3] = color_class1
    rgba[mask2, :3] = color_ambiguous

    # Set size 0 (rare) — dark gray
    mask0 = sizes == 0
    rgba[mask0, :3] = 0.2

    ax.scatter(
        grid_np[:, 0], grid_np[:, 1], c=rgba, s=1, rasterized=True
    )

    # Overlay training data
    X, y = np.asarray(data["X"]), np.asarray(data["label"])
    for c in range(2):
        ax.scatter(
            X[y == c, 0],
            X[y == c, 1],
            s=20,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.4,
            rasterized=False,
        )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=14)
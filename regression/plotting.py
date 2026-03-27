import shutil
from fileinput import filename

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from seaborn import colors

## Plotting setting
FONTSIZE = 18

# Try LaTeX settings, fall back gracefully
try:
    if shutil.which("latex") is not None:
        plt.rc("text", usetex=True)
        plt.rc(
            "text.latex",
            preamble=r"\usepackage{amsmath}\usepackage[bitstream-charter]{mathdesign}",
        )
        plt.rc("font", family="serif", size=FONTSIZE, serif="Charter")
    else:
        raise RuntimeError("LaTeX not found")
except Exception:
    # Fallback to default matplotlib serif font
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif", size=FONTSIZE)

plt.rc("axes", titlesize="medium")


def make_error_fig(
    data_train: list[np.ndarray],
    x_values: np.ndarray,
    y_mean: np.ndarray,
    y_std_epistemic: np.ndarray,
    y_std_aleatoric: np.ndarray,
    ci_lower_upper: list[np.ndarray] | None = None,
    CI: float = 0.95,
    network_name: str = "BNN",
    filename: str | None = None,
):
    colors = ["#1f77b4", "#d62728"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.02, 0.02, 0.98, 0.98))
    x_train, y_train = data_train
    y = 0.5 * np.sin(23 * x_values) + x_values / 2
    ax.scatter(x_train, y_train, s=10, alpha=1 / 16, color="k")
    ax.plot(x_values, y, color="black", label="True function", linewidth=1)
    ax.plot(x_values, y_mean, color=colors[1], label=f"{network_name} mean")
    z = norm.ppf((1 + CI) / 2)
    if ci_lower_upper is None:
        ax.fill_between(
            x_values,
            y_mean - z * y_std_epistemic,
            y_mean + z * y_std_epistemic,
            alpha=0.2,
            color=colors[1],
            label=rf"{int(CI*100)}\% interval (epistemic)",
        )
        ax.fill_between(
            x_values,
            y_mean - z * y_std_aleatoric,
            y_mean + z * y_std_aleatoric,
            alpha=0.2,
            color=colors[0],
            label=rf"{int(CI*100)}\% interval (aleatoric)",
        )
    else:
        ci_lower, ci_upper = ci_lower_upper
        ax.fill_between(
            x_values,
            ci_lower,
            ci_upper,
            alpha=0.2,
            color=colors[1],
            label=rf"{int(CI*100)}\% PI (total)",
        )
    ax.axvspan(-0.22, 0.22, color="gray", alpha=0.2)
    ax.axvspan(1, 1.2, color="gray", alpha=0.2)
    ax.axvspan(-1.2, -1.0, color="gray", alpha=0.2)
    ax.legend(fontsize=FONTSIZE - 5, loc="upper left")
    ax.set_ylim(-2, 2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_yticks([-2, -1, 0, 1, 2])
    if filename is not None:
        fig.savefig(f"{filename}.pdf")
    plt.show()
    plt.close()


def make_toy_fig(
    data_train: list[np.ndarray],
    x_values: np.ndarray,
    filename: str | None = None,
):
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.02, 0.02, 0.98, 0.98))
    x_train, y_train = data_train
    y = 0.5 * np.sin(23 * x_values) + x_values / 2
    ax.scatter(x_train, y_train, s=10, alpha=1 / 16, color="k")
    ax.plot(x_values, y, color="black", label="True function", linewidth=1)
    ax.axvspan(-0.22, 0.22, color="gray", alpha=0.2)
    ax.axvspan(1, 1.2, color="gray", alpha=0.2)
    ax.axvspan(-1.2, -1.0, color="gray", alpha=0.2)
    ax.legend(fontsize=FONTSIZE - 5, loc="upper left")
    ax.set_ylim(-2, 2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_yticks([-2, -1, 0, 1, 2])
    if filename is not None:
        fig.savefig(f"{filename}.pdf")
    plt.show()
    plt.close()


def make_pull_fig(
    x_values: list[np.ndarray],
    y_values: list[np.ndarray],
    y_means: list[np.ndarray],
    y_stds: list[np.ndarray],
    network_names: list[str] = ["BNN"],
    filename: str | None = None,
    title: str | None = "in-training domain",
):
    colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.08, 0.10, 0.98, 0.95))
    y_truth = 0.5 * np.sin(23 * x_values[0]) + x_values[0] / 2
    for i, (y_mean, y_std, network_name) in enumerate(
        zip(y_means, y_stds, network_names)
    ):
        pull = (y_mean - y_values[i]) / y_std
        hist, bin_edges = np.histogram(pull, bins=60, range=(-4, 4), density=True)
        ax.step(
            bin_edges[:-1],
            hist,
            where="post",
            label=f"{network_name}",
            color=colors[i],
            linewidth=2,
        )
    ax.plot(
        bin_edges[:-1],
        norm.pdf(bin_edges[:-1]),
        "--",
        color="black",
        label="Normal",
        linewidth=1,
    )
    ax.set_title(title, loc="right", fontsize=FONTSIZE - 2)
    ax.legend(
        fontsize=FONTSIZE - 2,
        loc="upper left",
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
    )
    ax.set_xlabel("Pull")
    ax.set_ylim(0, 0.45)
    ax.set_xlim(-4.2, 4.2)
    # ax.set_yticks([0, 0.5, 1, 1.5, 2])
    if filename is not None:
        fig.savefig(f"{filename}.pdf", transparent=True)
    plt.show()
    plt.close()


def make_calibration_fig(
    y_tests: list[np.ndarray],
    y_means: list[np.ndarray],
    y_stds: list[np.ndarray],
    network_names: list[str] = ["BNN"],
    filename: str | None = None,
    title: str | None = "in-training domain",
):
    colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]
    # Nominal confidence levels
    levels = np.linspace(0.05, 0.99, 50)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.08, 0.10, 0.98, 0.95))
    for i, (y_mean, y_std, y_test, network_name) in enumerate(
        zip(y_means, y_stds, y_tests, network_names)
    ):
        empirical_coverage = []
        for level in levels:
            z = norm.ppf(0.5 + level / 2.0)  # symmetric central interval
            lower = y_mean - z * y_std
            upper = y_mean + z * y_std
            covered = ((y_test >= lower) & (y_test <= upper)).mean()
            empirical_coverage.append(covered)
        empirical_coverage = np.array(empirical_coverage)
        ax.plot(
            levels, empirical_coverage, label=network_name, color=colors[i], linewidth=2
        )
    ax.plot(levels, levels, "--", label="Ideal", color="black", linewidth=1)
    ax.set_xlabel("Nominal confidence level")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title, loc="right", fontsize=FONTSIZE - 2)
    ax.legend(
        fontsize=FONTSIZE - 2,
        loc="upper left",
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
    )
    if filename is not None:
        fig.savefig(f"{filename}.pdf", transparent=True)
    plt.show()
    plt.close()

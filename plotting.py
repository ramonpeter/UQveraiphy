import shutil
from fileinput import filename

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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

# Define color cycle
# colors = [f"C{i}" for i in range(5)]
colors = ["#1f77b4", "#d62728"]


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
            label=rf"{int(CI*100)}\% CI (epistemic)",
        )
        ax.fill_between(
            x_values,
            y_mean - z * y_std_aleatoric,
            y_mean + z * y_std_aleatoric,
            alpha=0.2,
            color=colors[0],
            label=rf"{int(CI*100)}\% CI (aleatoric)",
        )
    else:
        ci_lower, ci_upper = ci_lower_upper
        ax.fill_between(
            x_values,
            ci_lower,
            ci_upper,
            alpha=0.2,
            color=colors[1],
            label=rf"{int(CI*100)}\% CI (total)",
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

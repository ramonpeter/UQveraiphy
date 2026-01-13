from fileinput import filename
import matplotlib.pyplot as plt
import numpy as np
import shutil
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
colors = [f"C{i}" for i in range(5)]


def make_error_fig(
    data_train: list[np.ndarray],
    x_values: np.ndarray,
    y_mean: np.ndarray,
    y_std_epistemic: np.ndarray,
    y_std_aleatoric: np.ndarray,
    ci_lower_uper: list[np.ndarray] | None = None,
    CI: float = 0.95,
    network_name: str = "BNN",
    filename: str | None = None,
):
    """Make a figure with training and validation error."""
    plt.figure(figsize=(8, 5))
    x_train, y_train = data_train
    plt.scatter(x_train, y_train, s=10, alpha=1 / 16, label="Data", color="k")
    plt.plot(x_values, y_mean, color="C0", label=f"{network_name} mean")
    z = norm.ppf((1 + CI) / 2)
    if ci_lower_uper is None:
        plt.fill_between(
            x_values,
            y_mean - z * y_std_epistemic,
            y_mean + z * y_std_epistemic,
            alpha=0.2,
            color="C0",
            label=rf"{int(CI*100)}\% CI (epistemic)",
        )

        plt.fill_between(
            x_values,
            y_mean - z * y_std_aleatoric,
            y_mean + z * y_std_aleatoric,
            alpha=0.2,
            color="C1",
            label=rf"{int(CI*100)}\% CI (aleatoric)",
        )
    else:
        ci_lower, ci_upper = ci_lower_uper
        plt.fill_between(
            x_values,
            ci_lower,
            ci_upper,
            alpha=0.2,
            color="C0",
            label=rf"{int(CI*100)}\% CI (conformal)",
        )
    y = 0.5 * np.sin(23 * x_values) + x_values / 2
    plt.axvspan(-0.22, 0.22, color="gray", alpha=0.2)
    plt.axvspan(1, 1.2, color="gray", alpha=0.2)
    plt.axvspan(-1.2, -1.0, color="gray", alpha=0.2)
    plt.plot(x_values, y, color="C2", label="True function", linewidth=1)
    plt.legend(fontsize=FONTSIZE - 6, loc="lower right")
    plt.ylim(-2, 2)
    plt.xlim(-1.2, 1.2)
    plt.show()
    if filename is not None:
        plt.savefig(f"{filename}.pdf")
    plt.close()

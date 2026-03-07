import sklearn.datasets
import jax.numpy as jnp
from jaxtyping import Array, Float


def gen_data(n_obs: int, noise: float, seed: int) -> dict[str, Array]:
    X, label = sklearn.datasets.make_moons(
        n_samples=n_obs, noise=noise, random_state=seed
    )

    # Remove points in the middle square
    center_x, center_y = X.mean(axis=0)
    half_width = 0.8

    mask = ~(
        (X[:, 0] > center_x - half_width)
        & (X[:, 0] < center_x + half_width)
        & (X[:, 1] > center_y - half_width)
        & (X[:, 1] < center_y + half_width)
    )

    X = X[mask]
    label = label[mask]

    return {"X": jnp.array(X), "label": jnp.array(label)}


def create_test_grid(
    x_range: tuple[float, float], y_range: tuple[float, float], resolution: float = 0.05
) -> Float[Array, "n 2"]:
    x = jnp.arange(x_range[0], x_range[1] + resolution, resolution)
    y = jnp.arange(y_range[0], y_range[1] + resolution, resolution)
    X, Y = jnp.meshgrid(x, y)
    return jnp.stack([X.ravel(), Y.ravel()], axis=1)

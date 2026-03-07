import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def rbf_with_median_heuristic(
    x: Float[Array, "n_x dim"], z: Float[Array, "n_z dim"]
) -> Float[Array, "n_x n_z"]:
    def median_heuristic(x: Array) -> Float[Array, ""]:
        sq_dist = jnp.sum(jnp.square(x[:, None] - x[None, :]), axis=-1)
        h = (
            jnp.median(sq_dist[jnp.triu_indices(x.shape[0], k=1)]) / jnp.log(x.shape[0])
            + 1e-8
        )
        return h

    h = median_heuristic(jnp.vstack([x, z]))
    rbf = lambda x, z: jnp.exp(-0.5 * jnp.sum(jnp.square(x - z)) / h)
    return jax.vmap(jax.vmap(rbf, (None, 0)), (0, None))(x, z)

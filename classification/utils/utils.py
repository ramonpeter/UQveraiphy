import chex
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Float, Int, Array


def binary_cross_entropy(
    logits: Float[Array, "batch"], labels: Int[Array, "batch"]
) -> Float[Array, ""]:
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


def ber_entropy(p: Float[Array, "..."]) -> Float[Array, "..."]:
    return -jnp.where((p > 0) & (p < 1), p * jnp.log(p) + (1 - p) * jnp.log(1 - p), 0.0)


@jax.jit
def uncertainty_decomposition(
    prob: Float[Array, "n samples"],
) -> tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "n"]]:
    chex.assert_rank(prob, 2)

    total = ber_entropy(prob.mean(1))
    aleatoric = jax.vmap(ber_entropy, in_axes=1, out_axes=1)(prob).mean(1)
    epistemic = total - aleatoric

    return total, aleatoric, epistemic

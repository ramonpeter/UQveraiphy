import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import jax.random as jrandom
from typing import Callable
import jax


## BNN Layer
class BayesLinear(eqx.Module):
    mu_weight: Array
    logsig_weight: Array
    use_bias: bool
    mu_bias: Array
    logsig_bias: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        if use_bias:
            w_key, b_key = jrandom.split(key)
        else:
            w_key = key
        self.mu_weight = jrandom.normal(w_key, (out_features, in_features))
        self.logsig_weight = 0.01 * jnp.ones((out_features, in_features))
        self.use_bias = use_bias
        if use_bias:
            self.mu_bias = jrandom.normal(b_key, (out_features,))
            self.logsig_bias = 0.01 * jnp.ones((out_features,))

    def __call__(self, x, key: PRNGKeyArray | None = None):
        mu_out = jnp.dot(self.mu_weight, x)

        # sampled output vs map
        if key is not None:
            if self.use_bias:
                w_key, b_key = jrandom.split(key)
                bias = self.mu_bias + jnp.exp(self.logsig_bias) * jrandom.normal(
                    b_key, self.mu_bias.shape
                )
            else:
                w_key = key
                bias = 0.0

            sig_out = jnp.dot(jnp.exp(self.logsig_weight), x)
            out = mu_out + sig_out * jrandom.normal(w_key, mu_out.shape) + bias
        else:
            out = mu_out
            if self.use_bias:
                out = out + self.mu_bias
        return out


class BNN(eqx.Module):
    layers: list
    activation: Callable | None
    final_activation: Callable | None

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation=jax.nn.relu,
        final_activation=jax.nn.sigmoid,
        use_bias: bool = True,
        use_final_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ) -> None:
        keys = jrandom.split(key, depth + 1)
        widths = [
            in_size,
        ] + depth * [width_size]
        layers = []
        for i in range(len(widths) - 1):
            layers += [
                BayesLinear(widths[i], widths[i + 1], use_bias=use_bias, key=keys[i])
            ]
        self.layers = layers + [
            BayesLinear(widths[-1], out_size, use_bias=use_final_bias, key=keys[-1])
        ]
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x: Array, key: PRNGKeyArray | None) -> Array:
        use_key = key is not None
        if use_key:
            keys = jrandom.split(key, len(self.layers))
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x, keys[i] if use_key else None))
        x = self.layers[-1](x, keys[-1] if use_key else None)
        if self.final_activation:
            x = self.final_activation(x)
        return x

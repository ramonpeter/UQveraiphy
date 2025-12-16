from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray


class BayesLinear(eqx.Module):
    """Bayesian Linear Layer with Gaussian weight distributions."""

    mu_weight: Float[Array, "out_features in_features"]
    logsig_weight: Float[Array, "out_features in_features"]
    mu_bias: Optional[Float[Array, "out_features"]]
    logsig_bias: Optional[Float[Array, "out_features"]]
    use_bias: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ) -> None:
        w_key, b_key = jrandom.split(key)

        # Initialize with Xavier/Glorot-like scaling
        scale = jnp.sqrt(2.0 / (in_features + out_features))
        self.mu_weight = scale * jrandom.normal(w_key, (out_features, in_features))
        self.logsig_weight = jnp.full((out_features, in_features), jnp.log(0.01))

        self.use_bias = use_bias
        if use_bias:
            self.mu_bias = jnp.zeros((out_features,))
            self.logsig_bias = jnp.full((out_features,), jnp.log(0.01))
        else:
            self.mu_bias = None
            self.logsig_bias = None

    def __call__(
        self,
        x: Float[Array, "*batch in_features"],
        key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "*batch out_features"]:
        mu_out = x @ self.mu_weight.T

        if key is not None:
            # Sample weights: w ~ N(mu, sigma^2)
            w_key, b_key = jrandom.split(key) if self.use_bias else (key, None)

            sigma_weight = jnp.exp(self.logsig_weight)
            weight_sample = self.mu_weight + sigma_weight * jrandom.normal(
                w_key, self.mu_weight.shape
            )
            out = x @ weight_sample.T

            if self.use_bias:
                sigma_bias = jnp.exp(self.logsig_bias)
                bias_sample = self.mu_bias + sigma_bias * jrandom.normal(
                    b_key, self.mu_bias.shape
                )
                out = out + bias_sample
        else:
            # MAP estimate (use mean)
            out = mu_out
            if self.use_bias:
                out = out + self.mu_bias

        return out


class BNN(eqx.Module):
    """Bayesian Neural Network with Gaussian weight distributions."""

    layers: list[BayesLinear]
    activation: Optional[Callable]
    final_activation: Optional[Callable]

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Optional[Callable] = jax.nn.relu,
        final_activation: Optional[Callable] = jax.nn.sigmoid,
        use_bias: bool = True,
        use_final_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ) -> None:
        keys = jrandom.split(key, depth + 1)

        # Build hidden layers
        layers = []
        layer_sizes = [in_size] + [width_size] * depth

        for i in range(depth):
            layers.append(
                BayesLinear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    use_bias=use_bias,
                    key=keys[i]
                )
            )

        # Output layer
        layers.append(
            BayesLinear(
                layer_sizes[-1],
                out_size,
                use_bias=use_final_bias,
                key=keys[-1]
            )
        )

        self.layers = layers
        self.activation = activation
        self.final_activation = final_activation

    def __call__(
        self,
        x: Float[Array, "*batch in_size"],
        key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "*batch out_size"]:
        if key is not None:
            keys = jrandom.split(key, len(self.layers))
        else:
            keys = [None] * len(self.layers)

        # Hidden layers with activation
        for layer, layer_key in zip(self.layers[:-1], keys[:-1]):
            x = layer(x, layer_key)
            if self.activation is not None:
                x = self.activation(x)

        # Output layer
        x = self.layers[-1](x, keys[-1])
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

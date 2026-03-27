from dataclasses import dataclass
from typing import Callable

import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class MLP(eqx.Module):
    model: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray, width: int = 32, depth: int = 2) -> None:
        self.model = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, x: Float[Array, "2"]) -> Float[Array, ""]:
        return self.model(x).squeeze()


class Ensemble(eqx.Module):
    n_member: int
    ensemble: eqx.Module

    def __init__(
        self,
        net_fun: Callable[[PRNGKeyArray], eqx.Module],
        n_member: int,
        key: PRNGKeyArray,
    ) -> None:
        self.n_member = n_member
        keys = jr.split(key, n_member)
        self.ensemble = eqx.filter_vmap(net_fun)(keys)

    def _get_member(self, idx: int) -> eqx.Module:
        arrays, static = eqx.partition(self.ensemble, eqx.is_array)
        indexed_arrays = jax.tree.map(lambda x: x[idx], arrays)
        return eqx.combine(indexed_arrays, static)

    def __getitem__(self, idx: int) -> eqx.Module:
        return self._get_member(idx)

    def __call__(self, x: Array) -> Array:
        return eqx.filter_vmap(lambda net, x: net(x), in_axes=(eqx.if_array(0), None))(
            self.ensemble, x
        )

    def __len__(self) -> int:
        return self.n_member


class VariationalLinear(eqx.Module):
    mu: Float[Array, "out_size in_size"]
    rho: Float[Array, "out_size in_size"]
    bias_mu: Float[Array, "out_size"]
    bias_rho: Float[Array, "out_size"]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int, key: PRNGKeyArray):
        k1, k2 = jr.split(key)
        lim = jnp.sqrt(2 / in_size)
        self.in_size, self.out_size = in_size, out_size
        self.mu = jr.uniform(k1, (out_size, in_size), minval=-lim, maxval=lim)
        self.rho = jnp.full((out_size, in_size), -5.0)
        self.bias_mu = jnp.zeros((out_size,))
        self.bias_rho = jnp.full((out_size,), -5.0)

    def __call__(
        self, x: Float[Array, "in_size"], key: PRNGKeyArray
    ) -> Float[Array, "out_size"]:
        k1, k2 = jr.split(key)
        sigma = jax.nn.softplus(self.rho)
        bias_sigma = jax.nn.softplus(self.bias_rho)
        w = self.mu + sigma * jr.normal(k1, self.mu.shape)
        b = self.bias_mu + bias_sigma * jr.normal(k2, self.bias_mu.shape)
        return w @ x + b

    def kl_divergence(self) -> Float[Array, ""]:
        def kl_diag_gaussians(mu, rho):
            sig = jax.nn.softplus(rho)
            return jnp.sum(0.5 * (mu**2 + sig**2 - 1.0 - 2.0 * jnp.log(sig)))

        return kl_diag_gaussians(self.mu, self.rho) + kl_diag_gaussians(
            self.bias_mu, self.bias_rho
        )


class VariationalBNN(eqx.Module):
    layers: list[VariationalLinear]

    def __init__(self, key: PRNGKeyArray, width: int = 32) -> None:
        keys = jr.split(key, 3)
        self.layers = [
            VariationalLinear(2, width, keys[0]),
            VariationalLinear(width, width, keys[1]),
            VariationalLinear(width, 1, keys[2]),
        ]

    def __call__(
        self, x: Float[Array, "in_size"], key: PRNGKeyArray
    ) -> Float[Array, ""]:
        keys = jr.split(key, len(self.layers))
        for i, layer in enumerate(self.layers[:-1]):
            x = jax.nn.relu(layer(x, keys[i]))
        return self.layers[-1](x, keys[-1]).squeeze()

    def total_kl(self) -> Float[Array, ""]:
        return sum(
            l.kl_divergence() for l in self.layers if isinstance(l, VariationalLinear)
        )

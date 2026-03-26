"""Train each classification model over N seeds and report average Brier score and ECE."""

import argparse
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import blackjax
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from utils.data import gen_data
from utils.kernel import rbf_with_median_heuristic
from utils.utils import binary_cross_entropy
from utils.diagnostics import brier_score, expected_calibration_error


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = 12345
    n_obs: int = 500
    noise: float = 0.35
    width: int = 32
    depth: int = 2
    n_member: int = 5
    wd: float = 1 / 500
    n_steps: int = 5_000
    lr: float = 1e-3
    num_samples: int = 500
    num_warmup: int = 500


cfg = Config()


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class MLP(eqx.Module):
    model: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray):
        self.model = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=cfg.width,
            depth=cfg.depth,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, x: Float[Array, "2"]) -> Float[Array, ""]:
        return self.model(x).squeeze()


class Ensemble(eqx.Module):
    n_member: int
    ensemble: eqx.Module

    def __init__(self, net_fun, n_member: int, key: PRNGKeyArray):
        self.n_member = n_member
        keys = jr.split(key, n_member)
        self.ensemble = eqx.filter_vmap(net_fun)(keys)

    def __call__(self, x: Array) -> Array:
        return eqx.filter_vmap(
            lambda net, x: net(x), in_axes=(eqx.if_array(0), None)
        )(self.ensemble, x)


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

    def __call__(self, x: Float[Array, "in_size"], key: PRNGKeyArray) -> Float[Array, "out_size"]:
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
    layers: list

    def __init__(self, key: PRNGKeyArray):
        keys = jr.split(key, 3)
        self.layers = [
            VariationalLinear(2, cfg.width, keys[0]),
            VariationalLinear(cfg.width, cfg.width, keys[1]),
            VariationalLinear(cfg.width, 1, keys[2]),
        ]

    def __call__(self, x: Float[Array, "batch in_size"], key) -> Float[Array, "batch out_size"]:
        keys = jr.split(key, len(self.layers))
        for i, layer in enumerate(self.layers[:-1]):
            x = jax.nn.relu(layer(x, keys[i]))
        return self.layers[-1](x, keys[-1]).squeeze()

    def total_kl(self):
        return sum(
            l.kl_divergence() for l in self.layers if isinstance(l, VariationalLinear)
        )


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------
def train_mlp(data, steps, key):
    model = MLP(key)
    optim = optax.adamw(cfg.lr, weight_decay=cfg.wd)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(params)

    X = data["X"]
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = data["label"]

    @jax.jit
    def train_loop(params, opt_state):
        def body(carry, _):
            params, opt_state = carry
            def loss_fn(params):
                model = eqx.combine(params, static)
                logits = jax.vmap(model)(X)
                return binary_cross_entropy(logits, y)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None
        (params, opt_state), _ = jax.lax.scan(body, (params, opt_state), None, length=steps)
        return params

    params = train_loop(params, opt_state)
    return eqx.combine(params, static)


def train_ensemble(data, steps, key, n_member=5):
    model = Ensemble(lambda k: MLP(k), n_member=n_member, key=key)
    optim = optax.adamw(cfg.lr, weight_decay=cfg.wd)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(params)

    X = data["X"]
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = data["label"]

    @jax.jit
    def train_loop(params, opt_state):
        def body(carry, _):
            params, opt_state = carry
            def loss_fn(params):
                model = eqx.combine(params, static)
                logits = eqx.filter_vmap(model)(X)
                return jax.vmap(binary_cross_entropy, (1, None))(logits, y).sum()
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None
        (params, opt_state), _ = jax.lax.scan(body, (params, opt_state), None, length=steps)
        return params

    params = train_loop(params, opt_state)
    return eqx.combine(params, static)


def train_repulsive_ensemble(data, steps, key, n_member=5):
    model = Ensemble(lambda k: MLP(k), n_member=n_member, key=key)
    optim = optax.adamw(cfg.lr, weight_decay=cfg.wd)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(params)

    X = data["X"]
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = data["label"]

    @jax.jit
    def train_loop(params, opt_state):
        def body(carry, _):
            params, opt_state = carry
            def loss_fn(params):
                model = eqx.combine(params, static)
                logits = eqx.filter_vmap(model)(X)
                bce = jax.vmap(binary_cross_entropy, (1, None))(logits, y)
                k = rbf_with_median_heuristic(bce, jax.lax.stop_gradient(bce))
                return bce.sum() + jnp.sum(
                    (k.sum(1) / jax.lax.stop_gradient(k).sum(1) - 1) / X.shape[0], 0
                )
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None
        (params, opt_state), _ = jax.lax.scan(body, (params, opt_state), None, length=steps)
        return params

    params = train_loop(params, opt_state)
    return eqx.combine(params, static)


def train_mfvi(data, steps, key):
    X, y = data["X"], data["label"]
    model = VariationalBNN(key)
    optim = optax.adam(cfg.lr)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optim.init(params)
    X_norm = (X - X.mean(0)) / (X.std(0) + 1e-6)

    @jax.jit
    def train_loop(params, opt_state, key):
        def body(carry, i):
            params, opt_state, key = carry
            key, subkey = jr.split(jr.fold_in(key, i))

            def loss_fn(params):
                model = eqx.combine(params, static)
                keys = jr.split(subkey, X_norm.shape[0])
                logits = jax.vmap(model)(X_norm, keys)
                log_likelihood = -jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))
                kl_term = model.total_kl() / X_norm.shape[0]
                return -log_likelihood + 0.3 * kl_term

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state, key), None
        (params, opt_state, key), _ = jax.lax.scan(body, (params, opt_state, key), jnp.arange(steps))
        return params

    params = train_loop(params, opt_state, key)
    return eqx.combine(params, static)


def run_hmc(data, key):
    model = MLP(key)
    params, static = eqx.partition(model, eqx.is_array)

    def log_joint(params):
        log_prior = sum(
            jnp.sum(jax.scipy.stats.norm.logpdf(p))
            for p in jax.tree_util.tree_leaves(params)
        )
        model = eqx.combine(params, static)
        logits = jax.vmap(model)(data["X"])
        log_lik = -jnp.sum(optax.sigmoid_binary_cross_entropy(logits, data["label"]))
        return log_prior + log_lik

    warmup_key, sample_key = jr.split(key)
    warmup = blackjax.window_adaptation(blackjax.nuts, log_joint)
    (state, kernel_params), _ = warmup.run(warmup_key, params, num_steps=cfg.num_warmup)
    nuts_kernel = blackjax.nuts(log_joint, **kernel_params)

    def inference_loop(state, key):
        state, info = nuts_kernel.step(key, state)
        return state, state.position

    keys = jr.split(sample_key, cfg.num_samples)
    _, samples = jax.lax.scan(inference_loop, state, keys)
    return samples, static


@partial(jax.jit, static_argnums=2)
def predict_hmc(X, samples, static):
    def single_pred(params):
        model = eqx.combine(params, static)
        return jax.nn.sigmoid(jax.vmap(model)(X))
    return jax.vmap(single_pred)(samples)


# ---------------------------------------------------------------------------
# Evaluation for a single seed
# ---------------------------------------------------------------------------
def evaluate_seed(seed):
    """Train all models with the given seed and return Brier/ECE on a held-out test set."""
    key = jr.PRNGKey(seed)

    # Generate training data
    data = gen_data(cfg.n_obs, cfg.noise, seed)
    mean, std = data["X"].mean(0), data["X"].std(0)

    # Held-out test set (fixed across seeds for fair comparison)
    test_data = gen_data(n_obs=1000, noise=cfg.noise, seed=seed + 10_000)
    X_test = test_data["X"]
    y_test = test_data["label"]
    X_test_norm = (X_test - mean) / (std + 1e-6)

    results = {}

    # --- Deterministic MLP ---
    key, subkey = jr.split(key)
    det_net = train_mlp(data, cfg.n_steps, subkey)

    @eqx.filter_jit
    def predict_det(model, X):
        return jax.nn.sigmoid(jax.vmap(model)(X))

    p = predict_det(det_net, X_test_norm)
    results["Deterministic"] = {
        "brier": float(brier_score(p, y_test)),
        "ece": float(expected_calibration_error(p, y_test, n_bins=20)),
    }

    # --- Ensemble ---
    key, subkey = jr.split(key)
    ens = train_ensemble(data, cfg.n_steps, subkey, n_member=cfg.n_member)

    @eqx.filter_jit
    def predict_ens(model, X):
        return jax.nn.sigmoid(jax.vmap(model)(X)).mean(1)

    p = predict_ens(ens, X_test_norm)
    results["Ensemble"] = {
        "brier": float(brier_score(p, y_test)),
        "ece": float(expected_calibration_error(p, y_test, n_bins=20)),
    }

    # --- Repulsive Ensemble ---
    key, subkey = jr.split(key)
    rep_ens = train_repulsive_ensemble(data, cfg.n_steps, subkey, n_member=cfg.n_member)
    p = predict_ens(rep_ens, X_test_norm)
    results["Rep. Ensemble"] = {
        "brier": float(brier_score(p, y_test)),
        "ece": float(expected_calibration_error(p, y_test, n_bins=20)),
    }

    # --- Mean-field VI ---
    key, subkey = jr.split(key)
    vi_model = train_mfvi(data, 2 * cfg.n_steps, subkey)
    vi_keys = jr.split(jr.PRNGKey(seed + 20_000), 500)

    @eqx.filter_jit
    def predict_vi(model, X, keys):
        return jax.nn.sigmoid(
            jax.vmap(lambda x: jax.vmap(lambda k: model(x, key=k))(keys))(X)
        ).mean(1)

    p = predict_vi(vi_model, X_test_norm, vi_keys)
    results["Mean-field VI"] = {
        "brier": float(brier_score(p, y_test)),
        "ece": float(expected_calibration_error(p, y_test, n_bins=20)),
    }

    # --- HMC ---
    key, subkey = jr.split(key)
    hmc_samples, hmc_static = run_hmc(data, subkey)
    p = predict_hmc(X_test, hmc_samples, hmc_static).mean(0)
    results["HMC"] = {
        "brier": float(brier_score(p, y_test)),
        "ece": float(expected_calibration_error(p, y_test, n_bins=20)),
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark classification models over multiple seeds.")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--start_seed", type=int, default=0, help="Starting seed value")
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    method_names = ["Deterministic", "Ensemble", "Rep. Ensemble", "Mean-field VI", "HMC"]

    all_results = {name: {"brier": [], "ece": []} for name in method_names}

    for i, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"Seed {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*50}")
        res = evaluate_seed(seed)
        for name in method_names:
            all_results[name]["brier"].append(res[name]["brier"])
            all_results[name]["ece"].append(res[name]["ece"])

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Results over {args.n_seeds} seeds ({args.start_seed}..{args.start_seed + args.n_seeds - 1})")
    print(f"{'='*60}")
    print(f"{'Method':20s}  {'Brier':>14s}  {'ECE':>14s}")
    print("-" * 54)
    for name in method_names:
        bs = np.array(all_results[name]["brier"])
        ece = np.array(all_results[name]["ece"])
        print(f"{name:20s}  {bs.mean():.4f} ± {bs.std():.4f}  {ece.mean():.4f} ± {ece.std():.4f}")


if __name__ == "__main__":
    main()

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import blackjax
from jaxtyping import Array, Float, PRNGKeyArray

from .config import Config
from .models import MLP, Ensemble, VariationalBNN
from .utils import binary_cross_entropy
from .kernel import rbf_with_median_heuristic


def train_mlp(
    data: dict[str, Array], steps: int, key: PRNGKeyArray, cfg: Config
) -> MLP:
    model = MLP(key, width=cfg.width, depth=cfg.depth)
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

        (params, opt_state), _ = jax.lax.scan(
            body, (params, opt_state), None, length=steps
        )
        return params

    params = train_loop(params, opt_state)
    return eqx.combine(params, static)


def train_ensemble(
    data: dict[str, Array],
    steps: int,
    key: PRNGKeyArray,
    cfg: Config,
    n_member: int = 5,
) -> Ensemble:
    model = Ensemble(
        lambda k: MLP(k, width=cfg.width, depth=cfg.depth),
        n_member=n_member,
        key=key,
    )
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

        (params, opt_state), _ = jax.lax.scan(
            body, (params, opt_state), None, length=steps
        )
        return params

    params = train_loop(params, opt_state)
    return eqx.combine(params, static)


def train_repulsive_ensemble(
    data: dict[str, Array],
    steps: int,
    key: PRNGKeyArray,
    cfg: Config,
    n_member: int = 5,
) -> Ensemble:
    model = Ensemble(
        lambda k: MLP(k, width=cfg.width, depth=cfg.depth),
        n_member=n_member,
        key=key,
    )
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

        (params, opt_state), _ = jax.lax.scan(
            body, (params, opt_state), None, length=steps
        )
        return params

    params = train_loop(params, opt_state)
    return eqx.combine(params, static)


def train_mfvi(
    data: dict[str, Array], steps: int, key: PRNGKeyArray, cfg: Config
) -> VariationalBNN:
    X, y = data["X"], data["label"]
    model = VariationalBNN(key, width=cfg.width)
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
                log_likelihood = -jnp.mean(
                    optax.sigmoid_binary_cross_entropy(logits, y)
                )
                kl_term = model.total_kl() / X_norm.shape[0]
                return -log_likelihood + 0.3 * kl_term

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state, key), None

        (params, opt_state, key), _ = jax.lax.scan(
            body, (params, opt_state, key), jnp.arange(steps)
        )
        return params

    params = train_loop(params, opt_state, key)
    return eqx.combine(params, static)


def run_hmc(data: dict[str, Array], key: PRNGKeyArray, cfg: Config) -> tuple[MLP, MLP]:
    model = MLP(key, width=cfg.width, depth=cfg.depth)
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
def predict_hmc(
    X: Float[Array, "n 2"], samples: MLP, static: MLP
) -> Float[Array, "num_samples n"]:
    def single_pred(params):
        model = eqx.combine(params, static)
        return jax.nn.sigmoid(jax.vmap(model)(X))

    return jax.vmap(single_pred)(samples)

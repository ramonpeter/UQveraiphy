import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from utils import bin_ce, pred_ensemble


def train_det_single(
    model: eqx.nn.MLP,
    data: dict[str, Float[Array, "n_samples ..."]],
    optim: optax.GradientTransformation,
    steps: int,
) -> eqx.nn.MLP:
    """Train a single deterministic network.

    Args:
        model: MLP network
        data: Dictionary with 'X' [n_samples, 2] and 'label' [n_samples]
        optim: Optimizer
        steps: Number of training steps

    Returns:
        Trained model
    """
    chex.assert_rank(data["X"], 2)
    chex.assert_rank(data["label"], 1)
    chex.assert_equal(data["X"].shape[0], data["label"].shape[0])

    @eqx.filter_jit
    def make_step(
        model: eqx.nn.MLP,
        opt_state: optax.OptState,
        x: Float[Array, "n 2"],
        y: Int[Array, "n"],
    ) -> tuple[eqx.nn.MLP, optax.OptState, Float[Array, ""], Float[Array, ""]]:
        def loss_fn(
            model: eqx.nn.MLP, x: Float[Array, "n 2"], y: Int[Array, "n"]
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            pred = jax.vmap(model)(x).squeeze()
            chex.assert_shape(pred, (x.shape[0],))
            pred = jnp.clip(pred, 1e-7, 1 - 1e-7)
            acc = jnp.mean((pred > 0.5) == y)
            return jnp.mean(bin_ce(pred, y)), acc

        (loss_val, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, x, y
        )
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, acc

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    for step in range(steps):
        model, opt_state, train_loss, accuracy = make_step(
            model, opt_state, data["X"], data["label"]
        )
        if step % 100 == 0:
            print(f"step={step}: train_loss={train_loss:.4f} / accuracy={accuracy:.2f}")

    return model


def train_det_net(
    model: eqx.nn.MLP,
    data: dict[str, Float[Array, "n_samples ..."]],
    optim: optax.GradientTransformation,
    steps: int,
) -> eqx.nn.MLP:
    """Train a single deterministic network.

    Args:
        model: MLP network
        data: Dictionary with 'X' [n_samples, 2] and 'label' [n_samples]
        optim: Optimizer
        steps: Number of training steps

    Returns:
        Trained model
    """
    chex.assert_rank(data["X"], 2)
    chex.assert_rank(data["label"], 1)
    chex.assert_equal(data["X"].shape[0], data["label"].shape[0])

    @eqx.filter_jit
    def make_step(
        model: eqx.nn.MLP,
        opt_state: optax.OptState,
        x: Float[Array, "n 2"],
        y: Int[Array, "n"],
    ) -> tuple[eqx.nn.MLP, optax.OptState, Float[Array, ""], Float[Array, ""]]:
        def loss_fn(
            model: eqx.nn.MLP, x: Float[Array, "n 2"], y: Int[Array, "n"]
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            pred = jax.vmap(model)(x).squeeze()
            chex.assert_shape(pred, (x.shape[0],))
            pred = jnp.clip(pred, 1e-7, 1 - 1e-7)
            acc = jnp.mean((pred > 0.5) == y)
            return jnp.mean(bin_ce(pred, y)), acc

        (loss_val, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, x, y
        )
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, acc

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    for step in range(steps):
        model, opt_state, train_loss, accuracy = make_step(
            model, opt_state, data["X"], data["label"]
        )
        if step % 100 == 0:
            print(f"step={step}: train_loss={train_loss:.4f} / accuracy={accuracy:.2f}")

    return model


def train_det_ens(
    model: eqx.nn.MLP,
    data: dict[str, Float[Array, "n_samples ..."]],
    optim: optax.GradientTransformation,
    steps: int,
) -> eqx.nn.MLP:
    """Train an ensemble of deterministic networks.

    Args:
        model: Ensemble of MLP networks (vmapped)
        data: Dictionary with 'X' [n_samples, 2] and 'label' [n_samples]
        optim: Optimizer
        steps: Number of training steps

    Returns:
        Trained ensemble
    """
    chex.assert_rank(data["X"], 2)
    chex.assert_rank(data["label"], 1)
    chex.assert_equal(data["X"].shape[0], data["label"].shape[0])

    @eqx.filter_jit
    def make_step(
        model: eqx.nn.MLP,
        opt_state: optax.OptState,
        x: Float[Array, "n 2"],
        y: Int[Array, "n"],
    ) -> tuple[eqx.nn.MLP, optax.OptState, Float[Array, ""], Float[Array, ""]]:
        def loss_fn(
            ens: eqx.nn.MLP, x: Float[Array, "n 2"], y: Int[Array, "n"]
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            # pred: [n_samples, n_ensemble]
            pred = jax.vmap(lambda xi: pred_ensemble(ens, xi))(x).squeeze()
            chex.assert_rank(pred, 2)
            chex.assert_equal(pred.shape[0], x.shape[0])

            pred = jnp.clip(pred, 1e-7, 1 - 1e-7)
            acc = jnp.mean((pred > 0.5) == y[:, None])

            # loss_per_member: [n_samples, n_ensemble]
            loss_per_member = jax.vmap(bin_ce, in_axes=(1, None), out_axes=1)(pred, y)
            chex.assert_shape(loss_per_member, pred.shape)

            # Average over batch, sum over ensemble members
            return jnp.sum(jnp.mean(loss_per_member, axis=0)), acc

        (loss_val, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, x, y
        )
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, acc

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    for step in range(steps):
        model, opt_state, train_loss, accuracy = make_step(
            model, opt_state, data["X"], data["label"]
        )
        if step % 100 == 0:
            print(f"step={step}: train_loss={train_loss:.4f} / accuracy={accuracy:.2f}")

    return model

def train_bnn(
    model,  # BNN type from bnn.py
    data: dict[str, Float[Array, "n_samples ..."]],
    optim: optax.GradientTransformation,
    steps: int,
    kl_weight: float = 0.01,
    n_samples: int = 1,
    *,
    key: PRNGKeyArray,
) -> tuple:  # Returns (BNN, PRNGKeyArray)
    """Train a Bayesian Neural Network with variational inference.

    Args:
        model: BNN network
        data: Dictionary with 'X' [n_samples, 2] and 'label' [n_samples]
        optim: Optimizer
        steps: Number of training steps
        kl_weight: Weight for KL divergence term
        n_samples: Number of samples for Monte Carlo estimation
        key: PRNG key

    Returns:
        Trained model and updated key
    """
    chex.assert_rank(data["X"], 2)
    chex.assert_rank(data["label"], 1)
    chex.assert_equal(data["X"].shape[0], data["label"].shape[0])

    def kl_divergence(model) -> Float[Array, ""]:
        """KL divergence between learned weights and standard normal prior."""
        kl = 0.0
        for layer in model.layers:
            # KL(q(w) || p(w)) for weights
            mu_w = layer.mu_weight
            logsig_w = layer.logsig_weight
            sig_w = jnp.exp(logsig_w)

            kl_w = 0.5 * jnp.sum(
                mu_w**2 + sig_w**2 - 2 * logsig_w - 1
            )
            kl += kl_w

            # KL for bias if present
            if layer.use_bias:
                mu_b = layer.mu_bias
                logsig_b = layer.logsig_bias
                sig_b = jnp.exp(logsig_b)

                kl_b = 0.5 * jnp.sum(
                    mu_b**2 + sig_b**2 - 2 * logsig_b - 1
                )
                kl += kl_b

        return kl

    @eqx.filter_jit
    def make_step(
        model,  # BNN
        opt_state: optax.OptState,
        x: Float[Array, "n 2"],
        y: Int[Array, "n"],
        key: PRNGKeyArray,
    ) -> tuple:  # (BNN, optax.OptState, Float, Float, PRNGKeyArray)
        def loss_fn(
            model, x: Float[Array, "n 2"], y: Int[Array, "n"], key: PRNGKeyArray
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            # Monte Carlo estimate of expected log likelihood
            keys = jrandom.split(key, n_samples)

            def single_sample_loss(k: PRNGKeyArray) -> Float[Array, ""]:
                pred = jax.vmap(lambda xi, ki: model(xi, ki))(
                    x, jrandom.split(k, x.shape[0])
                ).squeeze()
                chex.assert_shape(pred, (x.shape[0],))
                pred = jnp.clip(pred, 1e-7, 1 - 1e-7)
                return jnp.mean(bin_ce(pred, y))

            nll = jnp.mean(jax.vmap(single_sample_loss)(keys))

            # KL divergence
            kl = kl_divergence(model)

            # ELBO = -NLL - KL_weight * KL
            loss = nll + kl_weight * kl / x.shape[0]

            # Accuracy with MAP estimate
            pred_map = jax.vmap(lambda xi: model(xi, None))(x).squeeze()
            pred_map = jnp.clip(pred_map, 1e-7, 1 - 1e-7)
            acc = jnp.mean((pred_map > 0.5) == y)

            return loss, acc

        key, subkey = jrandom.split(key)
        (loss_val, acc), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, x, y, subkey
        )
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, acc, key

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    for step in range(steps):
        model, opt_state, train_loss, accuracy, key = make_step(
            model, opt_state, data["X"], data["label"], key
        )
        if step % 100 == 0:
            print(f"step={step}: train_loss={train_loss:.4f} / accuracy={accuracy:.2f}")

    return model, key

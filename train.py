import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int

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

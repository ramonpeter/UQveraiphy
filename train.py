import jax
import equinox as eqx
import jax.numpy as jnp
import optax

from utils import bin_ce, pred_ensemble


def train_det_single(
    model: eqx.nn.MLP, data: dict, optim: optax.GradientTransformation, steps: int
):

    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        def loss_fn(model, x, y):
            pred = jax.vmap(model)(x).squeeze()
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
            print(f"{step=}: {train_loss = :.4f} / {accuracy = :.2f}")


def train_det_net(
    model: eqx.nn.MLP, data: dict, optim: optax.GradientTransformation, steps: int
):

    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        def loss_fn(model, x, y):
            pred = jax.vmap(model)(x).squeeze()
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
            print(f"{step=}: {train_loss = :.4f} / {accuracy = :.2f}")

    return model


def train_det_ens(
    model: eqx.nn.MLP, data: dict, optim: optax.GradientTransformation, steps: int
):

    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        def loss_fn(ens, x, y):
            pred = jax.vmap(lambda x: pred_ensemble(ens, x))(
                data["X"]
            ).squeeze()  # [n_data x n_ens]
            pred = jnp.clip(pred, 1e-7, 1 - 1e-7)
            acc = jnp.mean((pred > 0.5) == y[:, None])
            loss_per_member = jax.vmap(bin_ce, in_axes=(1, None), out_axes=1)(
                pred, y
            )  # [n_data x n_ens]
            return (
                jnp.sum(jnp.mean(loss_per_member, axis=0)),
                acc,
            )  # avg over batch but not members

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
            print(f"{step=}: {train_loss = :.4f} / {accuracy = :.2f}")

    return model

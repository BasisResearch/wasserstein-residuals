"""Training utilities: optax optimiser loop with cosine schedule."""

import collections.abc

import equinox as eqx
import jax
import jax.tree_util as jtu
import optax
from tqdm import tqdm

from stitching._kde import SpatioTemporalData


def _fmt(v: float) -> str:
    """Format a scalar with fewer decimals for large values."""
    a = abs(v)
    if a < 0.001 and a > 0:
        return f"{v:.2e}"
    if a >= 1000:
        return f"{v:.1f}"
    if a >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


# Leaf names that should never receive optimizer updates even though
# they are JAX arrays.  These are structural / metadata arrays that
# participate in traced computations but are not model parameters.
_FROZEN_LEAF_NAMES: frozenset[str] = frozenset({"t_grid"})


def _trainable_filter(params, extra_frozen: tuple[str, ...] = ()):
    """Build a boolean pytree: True for trainable array leaves, False otherwise.

    Array leaves whose field name is in ``_FROZEN_LEAF_NAMES`` (or
    *extra_frozen*) are excluded so that the optimizer never updates them.
    """
    frozen = _FROZEN_LEAF_NAMES | frozenset(extra_frozen)
    flat, treedef = jtu.tree_flatten_with_path(params)
    bools = []
    for key_path, leaf in flat:
        if not eqx.is_array(leaf):
            bools.append(False)
            continue
        # Check if the last key in the path names a frozen field
        name = None
        if key_path:
            last = key_path[-1]
            if hasattr(last, "key"):  # GetAttrKey
                name = last.key
            elif hasattr(last, "name"):
                name = last.name
        bools.append(name not in frozen)
    return jtu.tree_unflatten(treedef, bools)


def train[M](
    params: M,
    loss_fn: collections.abc.Callable,
    data: SpatioTemporalData,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    has_aux: bool = True,
    key: jax.Array | None = None,
    epoch_callback: collections.abc.Callable[[M, int], None] | None = None,
    callback_every: int = 10,
    frozen_names: tuple[str, ...] = (),
    history: list[dict[str, float]] | None = None,
) -> M:
    """Train `params` by minimising `loss_fn(params, data_batch, key)`.

    Args:
        params: equinox module whose leaves are the trainable parameters.
        loss_fn: `(params, batch, key) -> (loss, aux)`.
        data: full training dataset.
        epochs: number of passes over the dataset.
        batch_size: minibatch size (samples are shuffled each epoch).
        lr: initial learning rate.
        has_aux: whether `loss_fn` returns a `(loss, aux_dict)` tuple.
        key: PRNG key.
        epoch_callback: optional `(model, epoch) -> None` called every
            *callback_every* epochs and at the final epoch.
        callback_every: how often (in epochs) to invoke the callback.

    Returns:
        Trained params.
    """
    _key: jax.Array = key if key is not None else jax.random.key(0)

    N = data.x.shape[0]
    batches_per_epoch = max(1, N // batch_size)
    total_steps = epochs * batches_per_epoch

    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=total_steps,
        alpha=0.05,
    )
    optimizer = optax.adam(learning_rate=schedule)
    filter_spec = _trainable_filter(params, extra_frozen=tuple(frozen_names))
    opt_state = optimizer.init(eqx.filter(params, filter_spec))

    # If loss_fn returns a plain scalar, wrap it so step always gets a tuple.
    _loss_fn = loss_fn
    if not has_aux:
        _raw = loss_fn

        def _loss_fn(params, batch, key):
            return (_raw(params, batch, key), {})

        has_aux = True

    @eqx.filter_jit
    def step(params, opt_state, batch, key):
        (loss, aux), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=has_aux)(
            params, batch, key
        )
        trainable_grads = eqx.filter(grads, filter_spec)
        trainable_params = eqx.filter(params, filter_spec)
        updates, opt_state_new = optimizer.update(
            trainable_grads, opt_state, trainable_params
        )
        params_new = eqx.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    iterator = tqdm(range(epochs), desc="train", leave=True)

    warmup_batch = SpatioTemporalData(
        x=data.x[:batch_size],
        t=data.t[:batch_size],
    )
    _key, warmup_key = jax.random.split(_key)
    _ = step(params, opt_state, warmup_batch, warmup_key)
    jax.block_until_ready(_)

    for epoch in iterator:
        _key, shuffle_key = jax.random.split(_key)
        perm = jax.random.permutation(shuffle_key, N)
        x_shuffled = data.x[perm]
        t_shuffled = data.t[perm]

        epoch_loss = 0.0
        epoch_aux: dict[str, float] = {}
        for b in range(batches_per_epoch):
            start = b * batch_size
            end = start + batch_size
            batch = SpatioTemporalData(x=x_shuffled[start:end], t=t_shuffled[start:end])

            _key, subkey = jax.random.split(_key)
            params, opt_state, loss, aux = step(params, opt_state, batch, subkey)
            epoch_loss += float(loss)
            if isinstance(aux, dict):
                for k, v in aux.items():
                    epoch_aux[k] = epoch_aux.get(k, 0.0) + float(v)

        epoch_loss /= batches_per_epoch
        epoch_aux = {k: v / batches_per_epoch for k, v in epoch_aux.items()}

        postfix = {"loss": _fmt(epoch_loss)}
        for k, v in epoch_aux.items():
            postfix[k] = _fmt(v)
        iterator.set_postfix(postfix)

        if history is not None:
            history.append({"epoch": epoch, "loss": epoch_loss, **epoch_aux})

        if epoch_callback is not None:
            is_last = epoch == epochs - 1
            if epoch % callback_every == 0 or is_last:
                epoch_callback(params, epoch)

    return params

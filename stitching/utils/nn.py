"""Equinox MLP with residual skip connections and SiLU activations."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp


def inv_softplus(y: float | jax.Array, beta: float = 1.0) -> jax.Array:
    """Inverse of softplus: x such that ``softplus(x) == y``.

    Accepts a Python scalar or a JAX array; returns a JAX array.
    Uses ``expm1`` for numerical stability at small *y*.
    """
    return jnp.log(jnp.expm1(beta * y)) / beta


class MLP(eqx.Module):
    """Feed-forward MLP with optional residual skip connections.

    Architecture:
    `Input -> [Linear -> SiLU (+ skip)] x (L-1) -> Linear -> Output`
    """

    layers: list[eqx.nn.Linear]
    use_skip: bool
    squeeze_output: bool
    nonnegative: bool

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        key: jax.Array,
        hidden_dims: tuple[int, ...] = (64, 64),
        use_skip: bool = True,
        squeeze_output: bool = False,
        nonnegative: bool = False,
        output_init_value: float | None = None,
    ):
        dims = [in_dim, *hidden_dims, out_dim]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        ]
        self.use_skip = use_skip
        self.squeeze_output = squeeze_output
        self.nonnegative = nonnegative

        # Optionally initialise the last layer so the network outputs a
        # constant value everywhere (useful for log-density networks).
        if output_init_value is not None:
            last = self.layers[-1]
            if nonnegative:
                raw = (
                    math.log(math.expm1(output_init_value))
                    if output_init_value > 0
                    else 0.0
                )
            else:
                raw = output_init_value
            assert last.bias is not None
            self.layers[-1] = eqx.tree_at(
                lambda layer: (layer.weight, layer.bias),
                last,
                (jnp.zeros_like(last.weight), jnp.full_like(last.bias, raw)),
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x
        for layer in self.layers[:-1]:
            z = jax.nn.silu(layer(y))
            y = y + z if self.use_skip and z.shape[-1] == y.shape[-1] else z
        y = self.layers[-1](y)

        if self.nonnegative:
            y = jax.nn.softplus(y)
        if self.squeeze_output:
            y = jnp.squeeze(y)
        return y


def field_mlp(
    dim: int,
    hidden: tuple[int, ...],
    key: jax.Array,
    *,
    time_conditioned: bool = False,
    output_init_value: float = 0.0,
) -> MLP:
    r"""Build an `MLP` representing a scalar field $f_\theta : \mathbb R^D \to \mathbb R$.

    Same architecture as `MLP`, with `out_dim=1` baked in and the last
    layer initialised so the field starts as the constant-*output_init_value*
    function (typically zero, or ``log(1/volume)`` for log-density nets).

    Pass ``time_conditioned=True`` to concatenate a time scalar onto the
    spatial input, giving a time-varying field $f_\theta(x, t)$.
    """
    return MLP(
        in_dim=dim + (1 if time_conditioned else 0),
        out_dim=1,
        hidden_dims=hidden,
        output_init_value=output_init_value,
        key=key,
    )

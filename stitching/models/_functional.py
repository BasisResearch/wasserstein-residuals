"""Shared energy-functional builder used by model `functional(...)` methods.

Each model exposes a `potential_net`, `entropy_coeff`, `potential_coeff` and
(optionally) interaction parameters; this module composes them into a
:class:`~stitching.velocity.WGFFunctional` consumed by the closed-form
Wasserstein-gradient seam.

`build_functional` is intentionally generic: it inspects whichever attributes
the model carries and includes the interaction term when an ``interaction_net``
is present (currently only `Stitching`).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from stitching.velocity import WGFFunctional


def build_functional(
    model: eqx.Module,
    *,
    freeze_entropy: float | None = None,
    freeze_potential: float | None = None,
    freeze_interaction: float | None = None,
    t: jax.Array | None = None,
) -> WGFFunctional:
    """Build the WGF energy functional from a model's potential/interaction nets.

    Each coefficient (entropy, potential, interaction) has two modes:
    - ``freeze_*=None`` (default): use the model's trainable (softplus)
      coefficient, optimised during training.
    - ``freeze_*=X``: fix the coefficient at scalar X (no gradient).

    When the model has a time-conditioned potential V_θ(x, t), pass the current
    ``t`` so the returned functional uses that snapshot's V.
    """
    is_time_cond = bool(getattr(model, "time_conditioned", False))
    if is_time_cond and t is not None:
        t_arr = jnp.atleast_1d(jnp.asarray(t))

        def potential(x):
            return jnp.squeeze(
                model.potential_net(jnp.concatenate([x, t_arr.astype(x.dtype)]))
            )
    else:

        def potential(x):
            return jnp.squeeze(model.potential_net(x))

    coeff_ent = (
        jnp.asarray(freeze_entropy)
        if freeze_entropy is not None
        else model.entropy_coeff
    )
    coeff_pot = (
        jnp.asarray(freeze_potential)
        if freeze_potential is not None
        else model.potential_coeff
    )

    W = None
    coeff_inter: jax.Array = jnp.asarray(0.0)
    interaction_net = getattr(model, "interaction_net", None)
    if interaction_net is not None:
        coeff_inter = (
            jnp.asarray(freeze_interaction)
            if freeze_interaction is not None
            else model.interaction_coeff
        )
        if model.interaction_type == "radial":

            def W(r) -> jax.Array:
                rsq = jnp.sum(r**2, axis=-1, keepdims=True)
                return jnp.squeeze(interaction_net(rsq))
        else:

            def W(r) -> jax.Array:
                return jnp.squeeze(interaction_net(r))

    return WGFFunctional(
        V=potential, W=W, c_H=coeff_ent, c_V=coeff_pot, c_W=coeff_inter
    )

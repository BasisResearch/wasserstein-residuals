"""Framework-free core: distribution protocol, type aliases, and `integrate`.

This is a self-contained, autodiff-free core — the type vocabulary the KDE
primitives and the closed-form velocity need, plus the ``integrate``
singledispatch (an expectation $\\mathbb{E}_\\rho[f]$, dispatched on the
distribution type). There is no generic Wasserstein autodiff machinery
(``Metric``/``Wasserstein.grad``/``first_variation``/``divergence``) in the
Stitching package — its gradient is closed form (see :mod:`stitching.velocity`).
"""

import collections.abc
import functools
import typing

import jax

type Time = jax.typing.ArrayLike
"""A time coordinate (scalar array-like)."""

type Scalar = jax.Array
"""A scalar JAX array."""


@typing.runtime_checkable
class Distribution[X](typing.Protocol):
    """A probability distribution over jax.Array samples."""

    def log_prob(self, x: X) -> Scalar: ...
    def sample(self, *, key: jax.Array, n: int) -> X: ...


type Functional[X] = collections.abc.Callable[[Distribution[X]], Scalar]
r"""An energy functional mapping a distribution to a scalar, $\mathcal{F}[\rho]$."""

type Curve[X] = collections.abc.Callable[[Time], Distribution[X]]
r"""A time-indexed curve of distributions, $t \mapsto \rho_t$."""

type Action[X] = collections.abc.Callable[[Curve[X]], Scalar]
"""An action functional mapping a curve of distributions to a scalar."""

type VectorField[X] = collections.abc.Callable[[X], X]
r"""A vector field $x \mapsto v(x)$ mapping `(D,)` to `(D,)`, e.g. velocity or score."""

type ScalarField[X] = collections.abc.Callable[[X], Scalar]
r"""A scalar field $x \mapsto f(x)$ mapping `(D,)` to `()`, e.g. potential or first variation."""

type TangentField[X, TX] = collections.abc.Callable[[X], TX]
r"""A tangent field $x \mapsto v(x)$ mapping `(D,)` to a tangent space, e.g. Wasserstein gradient."""


@functools.singledispatch
def integrate[X, Y](dist: Distribution[X], f: collections.abc.Callable[[X], Y]) -> Y:
    r"""Compute $\mathbb{E}_\rho[f(x)]$.

    Dispatches on the distribution type to choose the integration strategy.

    No default fallback is provided — register a specialisation via
    `@integrate.register(MyDist)` for each concrete distribution type.
    """
    raise NotImplementedError(
        f"integrate not implemented for {type(dist).__name__}; "
        "register a specialisation via @integrate.register"
    )

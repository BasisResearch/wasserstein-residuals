"""Regenerate the chiral McKean-Vlasov dataset.

Writes ``data/chiral/chiral-simulation.npz`` with the
trajectory of N = 500 particles under the 2D non-conservative chiral
kernel

    K(r) = ρ·(-∇W(r)) + ω·R₉₀(-∇W(r)),
    W(r) = C_r e^{-‖r‖/ℓ_r} - C_a e^{-‖r‖/ℓ_a},

with C_r=1, C_a=0.375, ℓ_r=0.5, ℓ_a=1.5, ρ=0.2, ω=1.5. Initial cloud:
two horizontal Gaussians stacked at y=±2.

Run with::

    python data/chiral/generate.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# data/chiral/generate.py → repo root is parents[2] (chiral, data, <root>).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

SEED, N = 0, 500
SNAP_TIMES = (0.0, 50.0, 100.0, 200.0, 400.0, 600.0, 800.0)


def chiral_kernel(
    C_r=1.0, C_a=0.375, ell_r=0.5, ell_a=1.5, omega=1.5, radial_scale=0.2
):
    def W(r):
        rho = jnp.sqrt(jnp.sum(r * r) + 1e-12)
        return C_r * jnp.exp(-rho / ell_r) - C_a * jnp.exp(-rho / ell_a)

    grad_W = jax.grad(lambda r: jnp.squeeze(W(r)))

    def K(r):
        g = grad_W(r)
        return -(radial_scale * g + omega * jnp.stack([-g[1], g[0]]))

    return K


def simulate(K, x0, t_save, dt_save=0.5, dt_inner=0.05):
    pair_K = jax.vmap(jax.vmap(K))
    n_inner = max(1, int(round(dt_save / dt_inner)))
    inner_dt = dt_save / n_inner

    @jax.jit
    def step(x):
        for _ in range(n_inner):
            x = (
                x
                + inner_dt
                * jnp.sum(pair_K(x[:, None, :] - x[None, :, :]), axis=1)
                / x.shape[0]
            )
        return x

    snaps = {0.0: np.asarray(x0)}
    x, t, j = x0, 0.0, 1
    while j < len(t_save):
        x = step(x)
        t += dt_save
        if abs(t - t_save[j]) < 0.5 * dt_save:
            snaps[float(t_save[j])] = np.asarray(x)
            j += 1
    return snaps


def main():
    k1, k2 = jax.random.split(jax.random.key(42))
    top = jnp.array([0.0, 2.0]) + jnp.array([1.5, 0.2]) * jax.random.normal(
        k1, (N // 2, 2)
    )
    bot = jnp.array([0.0, -2.0]) + jnp.array([1.5, 0.2]) * jax.random.normal(
        k2, (N // 2, 2)
    )
    snaps = simulate(chiral_kernel(), jnp.concatenate([top, bot]), SNAP_TIMES)

    perm = np.random.default_rng(SEED).permutation(N)
    n_tr = N // 2
    tr, te = perm[:n_tr], perm[n_tr:]
    train_x = np.concatenate([snaps[t][tr] for t in SNAP_TIMES]).astype(np.float32)
    test_x = np.concatenate([snaps[t][te] for t in SNAP_TIMES]).astype(np.float32)
    train_t = np.repeat(SNAP_TIMES, n_tr).astype(np.float32)
    test_t = np.repeat(SNAP_TIMES, N - n_tr).astype(np.float32)

    out = Path(__file__).resolve().parent / "chiral-simulation.npz"
    np.savez(
        out,
        train_x=train_x,
        train_t=train_t,
        test_x=test_x,
        test_t=test_t,
        snap_times=np.asarray(SNAP_TIMES),
    )
    print(f"saved {out}")
    print(f"train: {train_x.shape}, test: {test_x.shape}, snaps: {SNAP_TIMES}")


if __name__ == "__main__":
    main()

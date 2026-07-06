#!/usr/bin/env python
"""Export synthetic-2D paired & unpaired snapshots to jkonet-star's
``--load-from-file`` layout for the 6 sensitive potentials of
\\autoref{tab:synthetic2d_unpaired_ratio}.

Writes
  jkonet-star/data/{potential}_{paired,unpaired}_seed0/
        data.npy            (T*N, D) flat
        sample_labels.npy   (T*N,)   integer time labels
        args.txt            self-describing metadata

Run downstream (per dataset):
  cd ../jkonet-star
  .venv/bin/python data_generator.py --load-from-file <folder> \\
      --test-ratio 0.5 --n-gmm-components 10
  .venv/bin/python train.py --solver jkonet-star-potential --dataset <folder>
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from stitching.synthetic import (  # noqa: E402
    get_system,
    simulate_uncoupled,
    uniform_box,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

POTENTIALS_6 = [
    "flowers",
    "zigzag_ridge",
    "watershed",
    "ishigami",
    "friedman",
    "wavy_plateau",
]


def write_dataset(
    folder: Path, traj: np.ndarray, dt: float, potential: str, regime: str
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    T1, N, D = traj.shape
    flat = traj.reshape(T1 * N, D).astype(np.float32)
    labels = np.repeat(np.arange(T1, dtype=np.int64), N)
    np.save(folder / "data.npy", flat)
    np.save(folder / "sample_labels.npy", labels)
    with (folder / "args.txt").open("w") as f:
        f.write(f"potential={potential}\n")
        f.write("internal=none\n")
        f.write("beta=0.0\n")
        f.write("interaction=none\n")
        f.write(f"dt={dt}\n")
        f.write(f"# T+1={T1}, N={N}, D={D}, regime={regime}\n")
    print(f"  wrote {folder} ({flat.shape})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--jkonet-star-root", type=Path, default=_PROJECT_ROOT.parent / "jkonet-star"
    )
    p.add_argument("--n-particles", type=int, default=2000)
    p.add_argument("--n-timesteps", type=int, default=5)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--potentials", nargs="+", default=POTENTIALS_6)
    args = p.parse_args()

    for pot in args.potentials:
        sys_obj = get_system(pot)
        sys_obj = dataclasses.replace(sys_obj, init=uniform_box(-4.0, 4.0))

        # Paired: single trajectory of N particles.
        key = jax.random.key(args.seed)
        traj_p, _ = sys_obj.simulate(
            key=key,
            n_particles=args.n_particles,
            n_timesteps=args.n_timesteps,
            dt=args.dt,
            n_substeps=1,
        )
        traj_p = np.asarray(traj_p)

        # Unpaired: each snapshot is an independent fresh simulation.
        traj_u, _ = simulate_uncoupled(
            sys_obj,
            key=jax.random.key(args.seed + 12345),
            n_particles=args.n_particles,
            n_timesteps=args.n_timesteps,
            dt=args.dt,
            n_substeps=1,
        )
        traj_u = np.asarray(traj_u)

        for regime, traj in (("paired", traj_p), ("unpaired", traj_u)):
            folder = args.jkonet_star_root / "data" / f"{pot}_{regime}_seed{args.seed}"
            print(f"[{pot}] {regime}")
            write_dataset(folder, traj, args.dt, pot, regime)


if __name__ == "__main__":
    main()

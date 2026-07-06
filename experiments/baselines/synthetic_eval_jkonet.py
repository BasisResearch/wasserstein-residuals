#!/usr/bin/env python
"""Evaluate the JKOnet$^\\star_V$ runs of ``synthetic_run_jkonet.py`` on
held-out test particles using iJKOnet's three metrics:

  - one-step EMD     (Wasserstein-1 of predicted vs observed at t+1)
  - L^2-UVP          ($\\mathbb{E}\\|\\nabla V_\\theta - \\nabla V^\\star\\|^2 / \\mathrm{Var}(\\rho_k)$, in %)
  - Bd^2_{W_2}-UVP   (Bures-Wasserstein gap of Gaussian fits, in %)

Outputs a CSV table with one row per (potential, regime) so we can paste
into the LaTeX synthetic-2D unpaired table.
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
import time
from pathlib import Path

JKONET = Path(
    os.environ.get(
        "JKONET_STAR_DIR", Path(__file__).resolve().parents[3] / "jkonet-star"
    )
)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(JKONET))
sys.path.insert(0, str(REPO_ROOT))
os.chdir(JKONET)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Load the potentials module directly from its file, bypassing the
# stitching.synthetic package __init__ (it imports .simulate, which pulls in
# diffrax — not installed in jkonet-star's venv). potentials.py is
# self-contained — only needs jax + jnp.
import importlib.util as _ilu  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import ot  # noqa: E402
import yaml  # noqa: E402
from dataset import PopulationEvalDataset  # noqa: E402
from models import EnumMethod, get_model  # noqa: E402
from utils.sde_simulator import get_SDE_predictions  # noqa: E402

_p_path = REPO_ROOT / "stitching" / "synthetic" / "potentials.py"
_spec = _ilu.spec_from_file_location("_synth_pots", _p_path)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
POTENTIALS_2D = _mod.POTENTIALS_2D

POTENTIALS_6 = [
    "flowers",
    "zigzag_ridge",
    "watershed",
    "ishigami",
    "friedman",
    "wavy_plateau",
]


def load_state(folder: str):
    """Reconstruct the JKOnet*_V TrainState from saved params."""
    out_dir = REPO_ROOT / "results" / "synthetic-jkonet" / folder
    with (out_dir / "params.pkl").open("rb") as f:
        params_list = pickle.load(f)
    with open(JKONET / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(JKONET / "config-jkonet-extra.yaml") as f:
        config.update(yaml.safe_load(f))
    key = jax.random.PRNGKey(0)
    eval_ds = PopulationEvalDataset(
        key,
        folder,
        str(EnumMethod.JKO_NET_STAR_POTENTIAL),
        config["metrics"]["wasserstein_error"],
        "test_data",
    )
    model = get_model(
        EnumMethod.JKO_NET_STAR_POTENTIAL, config, eval_ds.data_dim, eval_ds.dt
    )
    fresh = list(model.create_state(jax.random.PRNGKey(0)))
    state = tuple(
        s.replace(params=jax.tree_util.tree_map(jnp.asarray, p))
        for s, p in zip(fresh, params_list)
    )
    return model, state, eval_ds, config


def metrics(model, state, eval_ds, V_truth_np):
    """Return (EMD_one_step, L2_UVP, BW2_UVP) averaged over snapshot pairs."""
    V_fn = model.get_potential(state)
    beta = model.get_beta(state)
    interaction = model.get_interaction(state)

    init_pp = jnp.asarray(eval_ds.trajectory[0])  # test particles at t=0
    predictions = get_SDE_predictions(
        str(EnumMethod.JKO_NET_STAR_POTENTIAL),
        eval_ds.dt,
        eval_ds.T,
        1,
        V_fn,
        beta,
        interaction,
        jax.random.PRNGKey(0),
        init_pp,
    )

    # One-step EMD: at each (t, t+1), pick an OT plan from observed t to
    # observed t+1, then forward one JKO step from those test points and
    # measure W1 vs the observed t+1 cloud. To match iJKOnet's protocol we
    # report the simpler full-rollout one-step pairs: predictions[i] vs
    # eval_ds.trajectory[i] for i=1..T.
    emd_per = []
    for i in range(1, eval_ds.T + 1):
        pred = np.asarray(predictions[i])
        obs = np.asarray(eval_ds.trajectory[i])
        a = np.ones(pred.shape[0]) / pred.shape[0]
        b = np.ones(obs.shape[0]) / obs.shape[0]
        M = np.sqrt(((pred[:, None] - obs[None, :]) ** 2).sum(-1))
        emd_per.append(float(ot.emd2(a, b, M)))
    emd_mean = float(np.mean(emd_per))

    # L^2-UVP gradient-level error.
    grad_V_true = jax.jit(jax.grad(V_truth_np))
    grad_V = jax.jit(jax.grad(lambda x: V_fn(x).squeeze()))
    uvp_per = []
    for i in range(1, eval_ds.T + 1):
        pts = np.asarray(eval_ds.trajectory[i]).astype(np.float32)
        gV = np.asarray(jax.vmap(grad_V)(jnp.asarray(pts)))
        gT = np.asarray(jax.vmap(grad_V_true)(jnp.asarray(pts)))
        var = float(np.var(eval_ds.trajectory[i - 1]))
        tau = float(eval_ds.dt)
        if var > 1e-12:
            uvp_per.append(tau**2 * float(np.mean(((gV - gT) ** 2).sum(-1))) / var)
    l2_uvp = 100.0 * float(np.mean(uvp_per))

    # Bd^2_{W_2}-UVP: Bures-Wasserstein of Gaussian fits of pred vs obs.
    bw_per = []
    for i in range(1, eval_ds.T + 1):
        x = np.asarray(predictions[i])
        y = np.asarray(eval_ds.trajectory[i])
        mx, my = x.mean(0), y.mean(0)
        Sx = np.cov(x.T) + 1e-6 * np.eye(x.shape[1])
        Sy = np.cov(y.T) + 1e-6 * np.eye(y.shape[1])
        wx, Vx = np.linalg.eigh(Sx)
        Sx_half = (Vx * np.sqrt(np.maximum(wx, 0))) @ Vx.T
        mid = Sx_half @ Sy @ Sx_half
        wm, _ = np.linalg.eigh(mid)
        bw2 = float(
            ((mx - my) ** 2).sum()
            + np.trace(Sx)
            + np.trace(Sy)
            - 2 * np.sqrt(np.maximum(wm, 0)).sum()
        )
        var = float(np.var(eval_ds.trajectory[i - 1]))
        if var > 1e-12:
            bw_per.append(bw2 / var)
    bw_uvp = 100.0 * float(np.mean(bw_per))

    return emd_mean, l2_uvp, bw_uvp


def main() -> None:
    rows = [["potential", "regime", "EMD", "L2_UVP", "BW2_UVP"]]
    for pot in POTENTIALS_6:
        V_truth_np = POTENTIALS_2D[pot]
        for regime in ("paired", "unpaired"):
            folder = f"{pot}_{regime}_seed0"
            t0 = time.time()
            model, state, eval_ds, _ = load_state(folder)
            emd, l2, bw = metrics(model, state, eval_ds, V_truth_np)
            rows.append([pot, regime, f"{emd:.4f}", f"{l2:.2f}", f"{bw:.2f}"])
            print(
                f"{pot:18s} {regime:9s}  EMD={emd:.4f}  "
                f"L2-UVP={l2:.2f}  BW2-UVP={bw:.2f}  ({time.time() - t0:.1f}s)"
            )

    out_csv = REPO_ROOT / "results" / "synthetic-jkonet" / "metrics.csv"
    with out_csv.open("w") as f:
        csv.writer(f).writerows(rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()

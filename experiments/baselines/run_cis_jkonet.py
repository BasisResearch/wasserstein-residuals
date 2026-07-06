#!/usr/bin/env python
r"""CIS JKOnet$^\star_{V,W}$ baseline — one file, three stages.

Trains and evaluates the upstream jkonet-star (full V+W+entropy) model on
the CIS dataset, as the comparison row for \autoref{tab:cis}. Three
subcommands run the whole pipeline:

    # 1. stitching venv: dump CIS into jkonet-star's load-from-file format
    python experiments/baselines/run_cis_jkonet.py export

    # 2. then preprocess with the upstream data_generator (manual):
    cd ../jkonet-star
    .venv/bin/python data_generator.py --load-from-file cis \
        --test-ratio 0.5 --n-gmm-components 10

    # 3. jkonet-star venv: train (saves params.pkl + V/W figure)
    ../jkonet-star/.venv/bin/python experiments/baselines/run_cis_jkonet.py train

    # 4. jkonet-star venv: forward-roll metrics for the table
    ../jkonet-star/.venv/bin/python experiments/baselines/run_cis_jkonet.py eval

Ground truth (CIS preset): V*(x) = 0.1(‖x‖²-4)², W*(r) = -2.2 exp(-r²).
The radial-interaction variant ``--radial`` matches the Stitching model's
$W_\theta(\|x{-}y\|^2)$ parametrisation; outputs land in
``results/cis-jkonet-full[-radial]/``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[2]
JKONET = Path(os.environ.get("JKONET_STAR_DIR", REPO_ROOT.parent / "jkonet-star"))


# --- ground-truth CIS dynamics + comparison metric ------------------------
def V_true(x):
    return 0.1 * ((x * x).sum(-1) - 4.0) ** 2


def W_true(r):
    return -2.2 * np.exp(-(r**2))


def pattern_r2(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float(((a * b).sum() / denom) ** 2) if denom > 1e-12 else float("nan")


def results_dir(radial):
    return (
        REPO_ROOT
        / "results"
        / ("cis-jkonet-full-radial" if radial else "cis-jkonet-full")
    )


# --- stage 1: export CIS into jkonet-star's format ------------------------
def cmd_export(args):
    """Runs in the *stitching* venv: read CIS npz, write npy + labels."""
    npz = np.load(REPO_ROOT / "data" / "cis" / "cis-simulation.npz")
    traj = npz["trajectories"].astype(np.float32)  # (T, N, D)
    times = npz["times"]
    T, N, D = traj.shape
    print(
        f"Loaded CIS: T={T}, N={N}, D={D}, t in [{int(times.min())}, {int(times.max())}]"
    )

    out_dir = JKONET / "data" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data.npy", traj.reshape(T * N, D))
    np.save(out_dir / "sample_labels.npy", np.repeat(np.arange(T, dtype=np.int64), N))
    with (out_dir / "args.txt").open("w") as f:
        f.write(
            "potential=cis_doublewell\ninternal=wiener\nbeta=0.045\n"
            "interaction=cis_gaussian\ndt=1.0\n"
            f"# T={T}, N={N}, D={D}\n"
        )

    print(f"Wrote {out_dir}/data.npy + sample_labels.npy")
    print("Next: preprocess with the upstream data_generator, then `train`:")
    print(f"  cd {JKONET}")
    print(
        f"  .venv/bin/python data_generator.py --load-from-file {args.name} "
        "--test-ratio 0.5 --n-gmm-components 10"
    )


# --- jkonet-star env setup (deferred: only train/eval need the repo) ------
def _enter_jkonet(radial):
    """chdir into jkonet-star, load merged config, build the eval dataset+model."""
    sys.path.insert(0, str(JKONET))
    os.chdir(JKONET)
    import jax
    import jax.numpy as jnp  # noqa: F401 — re-exported for callers
    import yaml
    from dataset import PopulationEvalDataset
    from models import EnumMethod, get_model

    config = {}
    for name in ("config.yaml", "config-jkonet-extra.yaml"):
        with open(JKONET / name) as f:
            config.update(yaml.safe_load(f))
    if radial:
        config.setdefault("energy", {}).setdefault("model", {})[
            "radial_interaction"
        ] = True
        print("Using radial interaction kernel (RadialMLP).")

    eval_ds = PopulationEvalDataset(
        jax.random.PRNGKey(0),
        "cis",
        str(EnumMethod.JKO_NET_STAR),
        config["metrics"]["wasserstein_error"],
        "test_data",
    )
    print(f"  dim={eval_ds.data_dim}, dt={eval_ds.dt}, T={eval_ds.T}")
    model = get_model(EnumMethod.JKO_NET_STAR, config, eval_ds.data_dim, eval_ds.dt)
    return jax, jnp, EnumMethod, config, eval_ds, model


# --- stage 3: train the baseline ------------------------------------------
def cmd_train(args):
    """Runs in the *jkonet-star* venv: replicate train.py, pickle params, plot V/W."""
    jax, jnp, EnumMethod, config, eval_ds, model = _enter_jkonet(args.radial)

    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from train import numpy_collate  # repo-local: needs sys.path set by _enter_jkonet

    out = results_dir(args.radial)
    out.mkdir(parents=True, exist_ok=True)

    state = model.create_state(jax.random.PRNGKey(args.seed))
    train_ds = model.load_dataset("cis")
    torch.manual_seed(args.seed)
    bsz = config["train"]["batch_size"]
    loader = DataLoader(
        train_ds,
        batch_size=bsz if bsz > 0 else len(train_ds),
        shuffle=True,
        collate_fn=numpy_collate,
    )
    train_step = jax.jit(model.train_step) if args.epochs > 1 else model.train_step

    print(f"Training for {args.epochs} epochs...")
    t0 = time.time()
    bar = tqdm(range(1, args.epochs + 1))
    for epoch in bar:
        loss = 0.0
        for sample in loader:
            l, state = train_step(state, sample)
            loss += float(l)
        bar.set_description(f"epoch {epoch} | loss {loss / max(len(loader), 1):.5f}")
    wall = time.time() - t0
    print(f"  done in {wall:.0f}s")

    params = jax.tree_util.tree_map(np.asarray, [s.params for s in state])
    with (out / "params.pkl").open("wb") as f:
        pickle.dump(params, f)
    print(f"  saved params to {out / 'params.pkl'}")

    # V on a grid, W on a radial slice, with R^2 vs ground truth.
    g = np.linspace(-4.5, 4.5, 100)
    XX, YY = np.meshgrid(g, g)
    grid = jnp.asarray(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=jnp.float32)
    V_l = np.array(jax.vmap(model.get_potential(state))(grid)).reshape(100, 100)
    V_l -= V_l.mean()
    V_t = np.array(V_true(np.stack([XX, YY], axis=-1)))
    V_t -= V_t.mean()
    r2_V = pattern_r2(V_l, V_t)

    r = np.linspace(0.01, 8.0, 400)
    diffs = jnp.asarray(np.stack([r, np.zeros_like(r)], axis=1), dtype=jnp.float32)
    W_l = np.asarray(jax.vmap(model.get_interaction(state))(diffs)).squeeze()
    W_t = W_true(r)
    band = (r >= 0.5) & (r <= 4.0)
    r2_W = pattern_r2(W_l[band], W_t[band])
    print(f"  pattern R^2(V) = {r2_V:.3f}   R^2(W) = {r2_W:.3f}")

    fig, axes = plt.subplots(
        2,
        1,
        layout="constrained",
        figsize=(4.3, 6.0),
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    axes[0].contourf(XX, YY, V_l, levels=18, cmap="RdBu_r")
    axes[0].contour(
        XX, YY, V_t, levels=6, colors="k", linewidths=0.6, linestyles="--", alpha=0.85
    )
    axes[0].set_title(
        rf"JKOnet$^\star_{{V,W}}$:  $V_\theta$ vs $V^\star$  ($R^2={r2_V:.2f}$)"
    )
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].set_box_aspect(1 / 1.2)
    axes[1].plot(r, W_l, c="tab:blue", lw=1.4, label=r"$W_\theta$")
    axes[1].plot(r, W_t, c="k", lw=1.0, ls="--", label=r"$W^\star$")
    axes[1].axvspan(0.5, 4.0, color="tab:gray", alpha=0.15, label="data range")
    axes[1].axhline(0, color="k", lw=0.4, alpha=0.5)
    axes[1].set_xlabel("$r$")
    axes[1].set_ylabel(r"$W(r)$")
    axes[1].set_title(rf"$W_\theta$ vs $W^\star$  ($R^2={r2_W:.2f}$)")
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    png = out / (
        "cis_jkonet_full_radial_functionals.png"
        if args.radial
        else "cis_jkonet_full_functionals.png"
    )
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved figure to {png}")

    with (out / "metrics.txt").open("w") as f:
        f.write(
            f"epochs={args.epochs}\nwall_s={wall:.1f}\nR2_V={r2_V:.3f}\nR2_W={r2_W:.3f}\n"
        )


# --- stage 4: forward-roll distributional metrics -------------------------
def _dist_metrics(x, y):
    """(EMD, W2, BW2, MMD) between two empirical clouds (matches our Stitching eval)."""
    import ot

    a = np.ones(x.shape[0]) / x.shape[0]
    b = np.ones(y.shape[0]) / y.shape[0]
    M = np.sqrt(((x[:, None] - y[None, :]) ** 2).sum(-1))
    emd = float(ot.emd2(a, b, M))
    w2 = float(np.sqrt(max(ot.emd2(a, b, M**2), 0.0)))
    mx, my = x.mean(0), y.mean(0)
    Sx = np.cov(x.T) + 1e-6 * np.eye(x.shape[1])
    Sy = np.cov(y.T) + 1e-6 * np.eye(y.shape[1])
    wx, Vx = np.linalg.eigh(Sx)
    Sx_half = (Vx * np.sqrt(np.maximum(wx, 0))) @ Vx.T
    wm, _ = np.linalg.eigh(Sx_half @ Sy @ Sx_half)
    bw2 = float(
        ((mx - my) ** 2).sum()
        + np.trace(Sx)
        + np.trace(Sy)
        - 2 * np.sqrt(np.maximum(wm, 0)).sum()
    )
    z = np.concatenate([x, y])
    pdist = np.sqrt(((z[:, None] - z[None, :]) ** 2).sum(-1))
    sigma = float(np.median(pdist[pdist > 0]))

    def k(u, v):
        return np.exp(-((u[:, None] - v[None, :]) ** 2).sum(-1) / (2 * sigma**2))

    mmd = float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())
    return emd, w2, bw2, mmd


def cmd_eval(args):
    """Runs in the *jkonet-star* venv: roll the JKO chain, compute table metrics."""
    out = results_dir(args.radial)
    print(f"Loading params from {out / 'params.pkl'}")
    with (out / "params.pkl").open("rb") as f:
        params_list = pickle.load(f)

    jax, jnp, EnumMethod, _, eval_ds, model = _enter_jkonet(args.radial)
    from utils.sde_simulator import get_SDE_predictions  # repo-local: needs sys.path

    fresh = list(model.create_state(jax.random.PRNGKey(0)))
    state = tuple(
        s.replace(params=jax.tree_util.tree_map(jnp.asarray, p))
        for s, p in zip(fresh, params_list)
    )
    beta = model.get_beta(state)
    print(f"  beta = {beta}")

    init_pp = jnp.asarray(eval_ds.trajectory[0])
    t0 = time.time()
    predictions = get_SDE_predictions(
        str(EnumMethod.JKO_NET_STAR),
        eval_ds.dt,
        eval_ds.T,
        1,
        model.get_potential(state),
        beta,
        model.get_interaction(state),
        jax.random.PRNGKey(0),
        init_pp,
    )
    print(f"  rolled chain in {time.time() - t0:.1f}s, shape {predictions.shape}")

    rows = [
        _dist_metrics(np.asarray(predictions[i]), np.asarray(eval_ds.trajectory[t]))
        for i, t in enumerate(sorted(eval_ds.trajectory.keys()))
    ]
    emd, w2, bw2, mmd = np.mean(rows, axis=0)
    print("\nJKOnet*_{V,W} on CIS held-out, full-chain forward roll:")
    print(f"  EMD={emd:.4f}  W2={w2:.4f}  BW^2={bw2:.4f}  MMD={mmd:.4f}")
    with (out / "fullchain_metrics.txt").open("w") as f:
        f.write(f"EMD={emd:.4f}\nW2={w2:.4f}\nBW2={bw2:.4f}\nMMD={mmd:.4f}\n")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="stage", required=True)

    e = sub.add_parser(
        "export", help="dump CIS into jkonet-star's load-from-file format"
    )
    e.add_argument("--name", default="cis", help="folder under jkonet-star/data/")
    e.set_defaults(func=cmd_export)

    t = sub.add_parser("train", help="train JKOnet*_{V,W}, save params + V/W figure")
    t.add_argument(
        "--epochs", type=int, default=300, help="plateaus by ~200; 300 is safe"
    )
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--radial", action="store_true", help="radial W_θ(‖x-y‖²) MLP")
    t.set_defaults(func=cmd_train)

    v = sub.add_parser("eval", help="forward-roll the chain, compute table metrics")
    v.add_argument("--radial", action="store_true")
    v.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

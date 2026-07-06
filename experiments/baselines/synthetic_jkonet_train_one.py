#!/usr/bin/env python
"""Train JKOnet$^\\star_V$ on a single synthetic dataset folder using the
jkonet-star library, then pickle the final params for offline evaluation.
Mirrors ``run_cis_jkonet.py train`` but for the V-only potential solver.
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(JKONET))
os.chdir(JKONET)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from dataset import PopulationEvalDataset  # noqa: E402
from models import EnumMethod, get_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from train import numpy_collate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", required=True, help="dataset folder under jkonet-star/data/"
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out", required=True, help="output dir for params.pkl + metrics.txt"
    )
    args = p.parse_args()

    with open(JKONET / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(JKONET / "config-jkonet-extra.yaml") as f:
        config.update(yaml.safe_load(f))

    key = jax.random.PRNGKey(args.seed)
    eval_ds = PopulationEvalDataset(
        key,
        args.dataset,
        str(EnumMethod.JKO_NET_STAR_POTENTIAL),
        config["metrics"]["wasserstein_error"],
        "test_data",
    )
    print(f"  dim={eval_ds.data_dim}, dt={eval_ds.dt}, T={eval_ds.T}")

    model = get_model(
        EnumMethod.JKO_NET_STAR_POTENTIAL, config, eval_ds.data_dim, eval_ds.dt
    )
    state = model.create_state(jax.random.PRNGKey(args.seed))
    train_ds = model.load_dataset(args.dataset)
    torch.manual_seed(args.seed)
    bsz = config["train"]["batch_size"]
    loader = DataLoader(
        train_ds,
        batch_size=bsz if bsz > 0 else len(train_ds),
        shuffle=True,
        collate_fn=numpy_collate,
    )
    train_step = jax.jit(model.train_step) if args.epochs > 1 else model.train_step

    print(f"  training {args.epochs} epochs")
    t0 = time.time()
    bar = tqdm(range(1, args.epochs + 1))
    for epoch in bar:
        loss = 0.0
        for sample in loader:
            l, state = train_step(state, sample)
            loss += float(l)
        bar.set_description(f"epoch {epoch} loss {loss / max(len(loader), 1):.5f}")
    wall = time.time() - t0
    print(f"  done in {wall:.1f}s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = jax.tree_util.tree_map(np.asarray, [s.params for s in state])
    with (out / "params.pkl").open("wb") as f:
        pickle.dump(params, f)
    with (out / "metrics.txt").open("w") as f:
        f.write(f"epochs={args.epochs}\n")
        f.write(f"wall_s={wall:.1f}\n")
        f.write(f"dataset={args.dataset}\n")
    print(f"  saved {out / 'params.pkl'}")


if __name__ == "__main__":
    main()

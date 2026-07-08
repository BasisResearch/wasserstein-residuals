#!/usr/bin/env python
"""Sequentially run JKOnet$^\\star_V$ on the 6 sensitive synthetic potentials
in both paired and unpaired regimes (12 runs total). Datasets must already
have been exported by ``synthetic_export_jkonet.py``.

For each (potential, regime):
  1. ``data_generator.py --load-from-file <folder>`` (preprocessing).
  2. ``train.py --solver jkonet-star-potential --dataset <folder>`` for
     ``--epochs`` epochs.
  3. Pickle the final TrainState parameters and write to results dir.

Distributional metrics + L^2-UVP + Bd^2 metrics are computed in a follow-up
``synthetic_eval_jkonet.py`` script that loads the saved params and
forward-rolls the JKO chain, identical protocol to our Stitching eval.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

# jkonet-star upstream checkout: set JKONET_STAR_DIR to a local clone of
# https://github.com/antonioterpin/jkonet-star (see the README's "Baseline checkouts").
JKONET = Path(
    os.environ.get(
        "JKONET_STAR_DIR", Path(__file__).resolve().parents[3] / "jkonet-star"
    )
)
REPO_ROOT = Path(__file__).resolve().parents[2]
PY = Path(os.environ.get("JKONET_STAR_PYTHON", JKONET / ".venv" / "bin" / "python"))

POTENTIALS_6 = [
    "flowers",
    "zigzag_ridge",
    "watershed",
    "ishigami",
    "friedman",
    "wavy_plateau",
]


def run_dataset(folder: str, epochs: int, gmm: int) -> None:
    """Preprocess + train JKOnet*_V on one dataset folder."""
    data_dir = JKONET / "data" / folder
    if not (data_dir / "data.npy").exists():
        raise FileNotFoundError(f"missing {data_dir}/data.npy")

    # 1) preprocessing (computes couplings + train/test split if not done).
    couplings_done = list(data_dir.glob("couplings_*.npy"))
    if not couplings_done:
        print(f"  [{folder}] preprocessing...")
        subprocess.check_call(
            [
                str(PY),
                "data_generator.py",
                "--load-from-file",
                folder,
                "--test-ratio",
                "0.5",
                "--n-gmm-components",
                str(gmm),
            ],
            cwd=JKONET,
        )

    # 2) Train JKOnet*-Potential. We instrument via a small wrapper that
    # also dumps state.params at the end. Reuse our run_cis_jkonet.py
    # pattern: a lightweight in-process train loop that pickles params.
    out_dir = REPO_ROOT / "results" / "synthetic-jkonet" / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "params.pkl").exists() and not args.force:
        print(f"  [{folder}] params already saved, skipping train")
        return

    log = out_dir / "train.log"
    print(f"  [{folder}] training...")
    t0 = time.time()
    with log.open("w") as f:
        subprocess.check_call(
            [
                str(PY),
                str(Path(__file__).resolve().parent / "synthetic_jkonet_train_one.py"),
                "--dataset",
                folder,
                "--epochs",
                str(epochs),
                "--out",
                str(out_dir),
            ],
            cwd=JKONET,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    print(f"  [{folder}] done in {time.time() - t0:.0f}s")


def main() -> None:
    global args
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="JKOnet$^\\star_V$ training epochs (the synthetic "
        "5-snapshot loss converges very fast).",
    )
    p.add_argument("--gmm", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--potentials", nargs="+", default=POTENTIALS_6)
    p.add_argument("--regimes", nargs="+", default=["paired", "unpaired"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    for pot in args.potentials:
        for regime in args.regimes:
            folder = f"{pot}_{regime}_seed{args.seed}"
            run_dataset(folder, epochs=args.epochs, gmm=args.gmm)


if __name__ == "__main__":
    main()

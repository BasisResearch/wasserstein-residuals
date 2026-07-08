#!/usr/bin/env python
"""Sequentially run iJKOnet$_V$ on the 6 sensitive synthetic potentials in
both paired and unpaired regimes (12 runs total). Reads CSV-style metrics
from iJKOnet's stdout and writes a single ``metrics.csv`` next to ours.

Datasets are symlinks under ``iJKOnet/data/<potential>_<regime>_seed0``
into the jkonet-star data dir (so preprocessing already done).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

# iJKOnet upstream checkout: set IJKONET_DIR to point to a local clone of
# https://github.com/iciemc/iJKOnet (see the README's "Baseline checkouts").
IJKO = Path(os.environ.get("IJKONET_DIR", Path.home() / "iJKOnet"))
PY = Path(os.environ.get("IJKONET_PYTHON", IJKO / ".venv" / "bin" / "python"))
REPO_ROOT = Path(__file__).resolve().parents[2]

POTENTIALS_6 = [
    "flowers",
    "zigzag_ridge",
    "watershed",
    "ishigami",
    "friedman",
    "wavy_plateau",
]


def parse_metrics(text: str) -> dict[str, float]:
    """Parse iJKOnet's stdout for the most recent EMD/BW2_UVP/MMD averages."""
    out: dict[str, float] = {}
    # Match lines like:
    # "Epoch N | EMD one step ahead: mean=0.0262, std=0.0002"
    pat = re.compile(r"Epoch \d+ \| (\w+(?: one step ahead|))\s*:\s*mean=([-+0-9.eE]+)")
    for m in pat.finditer(text):
        out[m.group(1).strip()] = float(m.group(2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--K", type=int, default=5, help="K from the iJKOnet paper (= n_timesteps)."
    )
    p.add_argument("--potentials", nargs="+", default=POTENTIALS_6)
    p.add_argument("--regimes", nargs="+", default=["paired", "unpaired"])
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "synthetic-ijkonet"))
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "metrics.csv"

    rows: list[list] = [
        [
            "potential",
            "regime",
            "EMD",
            "BW2_UVP",
            "MMD",
            "EMD_Tong",
            "MMD_DMSB",
            "wall_s",
        ]
    ]
    for pot in args.potentials:
        for regime in args.regimes:
            ds = f"{pot}_{regime}_seed{args.seed}"
            run_dir = out_root / ds
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"
            print(f"[{ds}] training {args.epochs} epochs ...", flush=True)
            t0 = time.time()
            with log_path.open("w") as f:
                subprocess.run(
                    [
                        str(PY),
                        "train.py",
                        "--solver",
                        "inverse-jkonet-potential",
                        "--dataset",
                        ds,
                        "--epochs",
                        str(args.epochs),
                        "--K",
                        str(args.K),
                        "--seed",
                        str(args.seed),
                    ],
                    cwd=IJKO,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            wall = time.time() - t0
            text = log_path.read_text()
            metrics = parse_metrics(text)
            row = [
                pot,
                regime,
                metrics.get("EMD one step ahead", float("nan")),
                metrics.get("BW2_UVP one step ahead", float("nan")),
                metrics.get("MMD one step ahead", float("nan")),
                metrics.get("EMD_Tong one step ahead", float("nan")),
                metrics.get("MMD_DMSB one step ahead", float("nan")),
                wall,
            ]
            rows.append(row)
            print(
                f"  done in {wall:.1f}s -- "
                f"EMD={row[2]}, BW2_UVP={row[3]}, MMD={row[4]}",
                flush=True,
            )

    with csv_path.open("w") as f:
        csv.writer(f).writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()

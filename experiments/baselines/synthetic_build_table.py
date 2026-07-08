#!/usr/bin/env python
"""Combine JKOnet$^\\star_V$, iJKOnet$_V$ and stitching numbers into one
absolute-metric comparison table (EMD, $L^2$-UVP, $\\mathrm{Bd}^2_{W_2}$-UVP)
covering paired and unpaired regimes on the 6 sensitive potentials.

Stitching numbers are hard-coded from ``tab:synthetic2d`` in the appendix.
JKOnet$^\\star_V$ numbers come from ``synthetic_eval_jkonet.py`` (our pipeline,
all 3 metrics). iJKOnet$_V$ numbers come from ``synthetic_run_ijkonet.py``
(parsed from upstream stdout: EMD and BW2 only -- L^2-UVP not emitted).
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

POTENTIALS_6 = [
    ("flowers", 1),
    ("zigzag_ridge", 4),
    ("watershed", 6),
    ("ishigami", 7),
    ("friedman", 8),
    ("wavy_plateau", 11),
]

# Stitching from tab:synthetic2d (appendix). (paired, unpaired) per metric.
STITCH = {
    "flowers": {"EMD": (0.30, 0.31), "L2": (0.00, 0.01), "BW2": (0.18, 0.24)},
    "zigzag_ridge": {"EMD": (0.29, 0.30), "L2": (0.03, 0.04), "BW2": (0.17, 0.29)},
    "watershed": {"EMD": (0.30, 0.30), "L2": (0.00, 0.01), "BW2": (0.19, 0.24)},
    "ishigami": {"EMD": (0.30, 0.30), "L2": (0.01, 0.01), "BW2": (0.17, 0.25)},
    "friedman": {"EMD": (0.29, 0.31), "L2": (0.06, 0.06), "BW2": (0.20, 0.27)},
    "wavy_plateau": {"EMD": (0.27, 0.29), "L2": (0.02, 0.04), "BW2": (0.51, 0.54)},
}


def read_csv(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Return {(potential, regime): {col: value}}."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            cols = {}
            for k, v in row.items():
                if k in ("potential", "regime"):
                    continue
                try:
                    cols[k] = float(v)
                except (ValueError, TypeError):
                    pass
            out[(row["potential"], row["regime"])] = cols
    return out


def fmt(v):
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    av = abs(v)
    if av < 0.01:
        return f"{v:.3f}"
    if av < 1:
        return f"{v:.2f}"
    if av < 100:
        return f"{v:.1f}"
    return f"{v:.0f}"


def main() -> None:
    jko = read_csv(REPO_ROOT / "results" / "synthetic-jkonet" / "metrics.csv")
    ijk = read_csv(REPO_ROOT / "results" / "synthetic-ijkonet" / "metrics.csv")

    def get(d, key, regime, col):
        v = d.get((key, regime), {}).get(col)
        return v if v is not None else float("nan")

    print("\\begin{table}[tb]")
    print("\\centering")
    print(
        "\\caption{\\textbf{Absolute-metric comparison on the $6$ "
        "potentials of \\citet[Tab.~3]{persiianov2025ijkonet} most "
        "sensitive to the paired$\\to$unpaired transition.} "
        "JKOnet$^\\star_V$ trained $100$ epochs through our eval pipeline; "
        "iJKOnet$_V$ trained $2{,}000$ epochs (their default), metrics from "
        "their stdout (no $L^2$-UVP available); Stitching from "
        "\\autoref{tab:synthetic2d}. All metrics lower is better; "
        "$L^2$-UVP and $\\mathrm{Bd}^2_{W_2}$-UVP in percent.}"
    )
    print("\\label{tab:synthetic2d_abs}")
    print("\\scriptsize")
    print("\\setlength{\\tabcolsep}{3pt}")
    print("\\adjustbox{max width=\\linewidth}{%")
    print("\\begin{tabular}{llrrr|rrr|rrr|rrr|rrr|rrr}")
    print("\\toprule")
    print(
        " & & \\multicolumn{9}{c|}{\\textbf{paired}} & "
        "\\multicolumn{9}{c}{\\textbf{unpaired}} \\\\"
    )
    print("\\cmidrule(lr){3-11} \\cmidrule(lr){12-20}")
    print(
        " & & \\multicolumn{3}{c|}{EMD} & \\multicolumn{3}{c|}{$L^2$-UVP} & "
        "\\multicolumn{3}{c|}{$\\mathrm{Bd}^2_{W_2}$-UVP} & "
        "\\multicolumn{3}{c|}{EMD} & \\multicolumn{3}{c|}{$L^2$-UVP} & "
        "\\multicolumn{3}{c}{$\\mathrm{Bd}^2_{W_2}$-UVP} \\\\"
    )
    methods = "J & I & S"
    print(
        f"\\# & potential & {methods} & {methods} & {methods} & "
        f"{methods} & {methods} & {methods} \\\\"
    )
    print("\\midrule")

    for pot, idx in POTENTIALS_6:
        st = STITCH[pot]
        cells = []
        for regime in ("paired", "unpaired"):
            r_idx = 0 if regime == "paired" else 1
            # EMD
            j_emd = get(jko, pot, regime, "EMD")
            i_emd = get(ijk, pot, regime, "EMD")
            s_emd = st["EMD"][r_idx]
            # L2-UVP
            j_l2 = get(jko, pot, regime, "L2_UVP")
            i_l2 = float("nan")  # not in iJKO csv
            s_l2 = st["L2"][r_idx]
            # BW2-UVP -- both JKO* and iJKO emit values in % already.
            j_bw = get(jko, pot, regime, "BW2_UVP")
            i_bw = get(ijk, pot, regime, "BW2_UVP")
            s_bw = st["BW2"][r_idx]
            cells += [j_emd, i_emd, s_emd, j_l2, i_l2, s_l2, j_bw, i_bw, s_bw]
        # bold the best non-NaN among each triple
        out_strs = []
        for k in range(0, len(cells), 3):
            triple = cells[k : k + 3]
            valid = [(v, j) for j, v in enumerate(triple) if v == v]
            best_j = min(valid)[1] if valid else None
            for j, v in enumerate(triple):
                s = fmt(v)
                if j == best_j and s != "--":
                    out_strs.append(f"$\\mathbf{{{s}}}$")
                else:
                    out_strs.append(f"${s}$")
        row = (
            f"{idx:>2} & {pot.replace('_', '\\_')} & " + " & ".join(out_strs) + " \\\\"
        )
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print(
        "\\\\\\smallskip\\textit{\\scriptsize J = JKOnet$^\\star_V$, "
        "I = iJKOnet$_V$, S = Stitching. "
        "iJKOnet$_V$ does not expose $L^2$-UVP through its CLI.}"
    )
    print("\\end{table}")

    # Plain-text summary for the terminal.
    print("\n\n# Markdown summary (paired / unpaired)\n")
    for pot, _ in POTENTIALS_6:
        st = STITCH[pot]
        for regime in ("paired", "unpaired"):
            r = 0 if regime == "paired" else 1
            j_emd = get(jko, pot, regime, "EMD")
            i_emd = get(ijk, pot, regime, "EMD")
            s_emd = st["EMD"][r]
            j_l2 = get(jko, pot, regime, "L2_UVP")
            s_l2 = st["L2"][r]
            j_bw = get(jko, pot, regime, "BW2_UVP")
            i_bw = get(ijk, pot, regime, "BW2_UVP")
            s_bw = st["BW2"][r]
            print(
                f"| {pot:14s} | {regime:8s} | "
                f"EMD: J={fmt(j_emd):>6} I={fmt(i_emd):>6} S={fmt(s_emd):>6} | "
                f"L2: J={fmt(j_l2):>7} S={fmt(s_l2):>6} | "
                f"BW2: J={fmt(j_bw):>6} I={fmt(i_bw):>6} S={fmt(s_bw):>6} |"
            )


if __name__ == "__main__":
    main()

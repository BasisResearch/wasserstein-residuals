#!/usr/bin/env python
"""Build the OT-vs-McCann trajectory-init ablation write-up (LaTeX appendix).

Companion to :mod:`synthetic_build_table` -- same conventions (the 6 sensitive
potentials, the three absolute metrics EMD / $L^2$-UVP / $\\mathrm{Bd}^2_{W_2}$-UVP,
:func:`fmt` precision, bold-the-best), but comparing the two trajectory
*initializations* (OT vs.\\ McCann) of Stitching rather than competing methods.

Reads the metric CSVs written by the ``<exp>`` (OT baseline) and ``<exp>_mccann``
experiments and writes ``docs/mccann_ablation.tex``:

* Table 1 -- synthetic gallery, 6 potentials x paired/unpaired x {OT, McCann}.
* Table 2 -- EB (RNA) gappy leave-two-out benchmark, per-timepoint + mean.
* Table 3 -- CIS held-out distributional metrics.

Run from the repo root after both result trees exist::

    uv run python experiments/baselines/mccann_ablation_table.py
"""

from __future__ import annotations

import csv
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
DOCS = REPO_ROOT / "docs"
FIGDIR = DOCS / "figures"
OUT = DOCS / "mccann_ablation.tex"

# The 6 potentials most sensitive to the paired->unpaired transition, matching
# synthetic_build_table.POTENTIALS_6 (paper Tab. tab:synthetic2d_abs).
POTENTIALS_6 = ["flowers", "zigzag_ridge", "watershed", "ishigami", "friedman", "wavy_plateau"]
PRETTY = {
    "flowers": "Flowers",
    "zigzag_ridge": "Zigzag ridge",
    "watershed": "Watershed",
    "ishigami": "Ishigami",
    "friedman": "Friedman",
    "wavy_plateau": "Wavy plateau",
}
# The three absolute metrics reported in the paper, and their CSV columns.
METRICS = [("EMD", "test_emd"), (r"$L^2$-UVP", "l2_uvp"), (r"$\mathrm{Bd}^2_{W_2}$-UVP", "bw2_uvp")]


def fmt(v) -> str:
    """Adaptive precision matching synthetic_build_table.fmt."""
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


def _load(path: Path, key) -> dict:
    with path.open() as f:
        return {key(r): r for r in csv.DictReader(f)}


def _bold_lower(ot: float, mc: float) -> tuple[str, str]:
    """Format an (OT, McCann) pair as math, bolding the smaller (better) value."""
    ot_s, mc_s = fmt(ot), fmt(mc)
    if ot <= mc:
        return rf"$\mathbf{{{ot_s}}}$", f"${mc_s}$"
    return f"${ot_s}$", rf"$\mathbf{{{mc_s}}}$"


def _bold_higher(ot: float, mc: float) -> tuple[str, str]:
    """Format an (OT, McCann) pair as math, bolding the larger (better) value."""
    ot_s, mc_s = fmt(ot), fmt(mc)
    if ot >= mc:
        return rf"$\mathbf{{{ot_s}}}$", f"${mc_s}$"
    return f"${ot_s}$", rf"$\mathbf{{{mc_s}}}$"


def _load_multi(path: Path, key) -> dict[str, list[dict]]:
    """Group CSV rows by ``key`` (e.g. one entry per variant, many seeds)."""
    out: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            out[key(r)].append(r)
    return out


def _agg(rows: list[dict], col: str) -> tuple[float, float]:
    """Across-seed (mean, sample-std) of a numeric column, ignoring NaNs."""
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _fmt_pm(mean: float, std: float, *, bold: bool = False) -> str:
    """Render ``mean ± std`` in math mode (std suppressed when 0/NaN)."""
    if mean != mean:
        return "--"
    core = rf"\mathbf{{{fmt(mean)}}}" if bold else fmt(mean)
    if std == std and std > 0:
        return rf"${core}{{\scriptstyle\,\pm\,{fmt(std)}}}$"
    return rf"${core}$"


def _cell_pm(
    mo: float, so: float, mm: float, sm: float, *, lower_better: bool = True
) -> tuple[str, str]:
    """Format an (OT, McCann) mean±std pair, bolding the better mean."""
    if mo != mo or mm != mm:
        return _fmt_pm(mo, so), _fmt_pm(mm, sm)
    ot_better = (mo <= mm) if lower_better else (mo >= mm)
    return _fmt_pm(mo, so, bold=ot_better), _fmt_pm(mm, sm, bold=not ot_better)


def _synth() -> tuple[dict, dict]:
    ot = _load(RESULTS / "synthetic" / "metrics" / "synthetic_panel_summary.csv",
               lambda r: (r["potential"], r["mode"]))
    mc = _load(RESULTS / "synthetic_mccann" / "metrics" / "synthetic_panel_summary.csv",
               lambda r: (r["potential"], r["mode"]))
    return ot, mc


def table1(ot: dict, mc: dict) -> str:
    rows = []
    for pot in POTENTIALS_6:
        cells = []
        for mode in ("paired", "unpaired"):
            for _, col in METRICS:
                o, m = float(ot[(pot, mode)][col]), float(mc[(pot, mode)][col])
                cells.extend(_bold_lower(o, m))
        rows.append(f"{PRETTY[pot]:14s} & " + " & ".join(cells) + r" \\")
    body = "\n    ".join(rows)
    metric_hdr = " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}$\downarrow$}}" for name, _ in METRICS)
    return rf"""\begin{{table}}[tb]
  \centering
  \caption{{\textbf{{Trajectory-initialization ablation on the 6 sensitive
    synthetic potentials.}} Stitching trained identically (seed 0, 2000 epochs);
    only the seeding differs -- OT (Hungarian match + linear interpolation, O)
    vs.\ McCann (Bures--Wasserstein geodesic, M). All metrics lower is better;
    $L^2$-UVP and $\mathrm{{Bd}}^2_{{W_2}}$-UVP as in \autoref{{tab:synthetic2d_abs}}.
    Better initialization per cell in bold.}}
  \label{{tab:mccann-synthetic}}
  \scriptsize
  \setlength{{\tabcolsep}}{{4pt}}
  \adjustbox{{max width=\linewidth}}{{%
  \begin{{tabular}}{{l rr rr rr rr rr rr}}
    \toprule
    & \multicolumn{{6}}{{c}}{{\textbf{{paired}}}} & \multicolumn{{6}}{{c}}{{\textbf{{unpaired}}}} \\
    \cmidrule(lr){{2-7}}\cmidrule(lr){{8-13}}
    & {metric_hdr} & {metric_hdr} \\
    Potential & O & M & O & M & O & M & O & M & O & M & O & M \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}%
  }}
\end{{table}}"""


DISP = {"static": r"Static $V_\theta$", "tcond": r"Time-cond.\ $V_\theta$"}


def _pm_block(ot_rows: dict, mc_rows: dict, cols) -> tuple[str, int]:
    """Two rows (OT / McCann) of mean±std cells per variant; return body + seed count."""
    rows, n_seeds = [], 0
    for var in ("static", "tcond"):
        o, m = ot_rows[var], mc_rows[var]
        n_seeds = max(n_seeds, len(o), len(m))
        ot_cells, mc_cells = [], []
        for col in cols:
            oc, mcell = _cell_pm(*_agg(o, col), *_agg(m, col), lower_better=True)
            ot_cells.append(oc)
            mc_cells.append(mcell)
        rows.append(rf"\multirow{{2}}{{*}}{{{DISP[var]}}} & OT     & " + " & ".join(ot_cells) + r" \\")
        rows.append(r"                                  & McCann & " + " & ".join(mc_cells) + r" \\")
        rows.append(r"\addlinespace")
    if rows and rows[-1] == r"\addlinespace":
        rows.pop()  # no inter-block spacer after the final row
    return "\n    ".join(rows), n_seeds


def _rna_full(tree: str) -> dict[str, list[dict]]:
    """Full-data EB rows per variant, with a derived transition-mean column."""
    by_var = _load_multi(RESULTS / tree / "metrics" / "rna_metrics.csv", lambda r: r["variant"])
    for rows in by_var.values():
        for r in rows:
            ts = [float(r[f"w1_t{t}"]) for t in (1, 2, 3, 4)]
            r["w1_trans"] = str(sum(ts) / len(ts))
    return by_var


def table_eb_full() -> str:
    ot, mc = _rna_full("rna"), _rna_full("rna_mccann")
    cols = ["w1_t1", "w1_t2", "w1_t3", "w1_t4", "w1_trans"]
    body, n = _pm_block(ot, mc, cols)
    hdr = " & ".join(
        rf"${core}\downarrow$"
        for core in ("W_1(t_1)", "W_1(t_2)", "W_1(t_3)", "W_1(t_4)", r"\mathrm{mean}\,W_1")
    )
    return rf"""\begin{{table}}[tb]
  \centering
  \caption{{\textbf{{EB scRNA-seq full-data interpolation.}} Held-out $W_1$ at each
    observed timepoint $t_1,\dots,t_4$ and their mean (the initial condition $t_0$
    is excluded), OT vs.\ McCann initialization; ${n}$ seeds, mean${{\pm}}$std.
    Lower is better; better init per column in bold.}}
  \label{{tab:mccann-eb-full}}
  \setlength{{\tabcolsep}}{{4pt}}
  \adjustbox{{max width=\linewidth}}{{%
  \begin{{tabular}}{{ll ccccc}}
    \toprule
    Variant & Init & {hdr} \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}%
  }}
\end{{table}}"""


def table_eb_gappy() -> str:
    ot = _load_multi(RESULTS / "rna" / "metrics" / "rna_gappy_summary.csv", lambda r: r["variant"])
    mc = _load_multi(RESULTS / "rna_mccann" / "metrics" / "rna_gappy_summary.csv", lambda r: r["variant"])
    body, n = _pm_block(ot, mc, ["w2_t1", "w2_t3", "w2_mean"])
    return rf"""\begin{{table}}[tb]
  \centering
  \caption{{\textbf{{EB scRNA-seq gappy leave-two-out interpolation.}} $W_2$ at
    the two held-out timepoints ($t_1,t_3$) and their mean, OT vs.\ McCann
    initialization; ${n}$ seeds, mean${{\pm}}$std. Lower is better; better init
    per column in bold.}}
  \label{{tab:mccann-eb}}
  \begin{{tabular}}{{ll ccc}}
    \toprule
    Variant & Init & $W_2(t_1)\downarrow$ & $W_2(t_3)\downarrow$ & mean $W_2\downarrow$ \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}"""


def table_cis() -> str:
    o = _load(RESULTS / "cis" / "metrics" / "metrics.csv", lambda r: r["variant"])["cis"]
    m = _load(RESULTS / "cis_mccann" / "metrics" / "metrics.csv", lambda r: r["variant"])["cis"]
    emd = _bold_lower(float(o["emd_w1"]), float(m["emd_w1"]))
    w2 = _bold_lower(float(o["w2"]), float(m["w2"]))
    bw2 = _bold_lower(float(o["bw2"]), float(m["bw2"]))
    mmd = _bold_lower(float(o["mmd"]), float(m["mmd"]))
    r2v = _bold_higher(float(o["r2_v"]), float(m["r2_v"]))
    r2w = _bold_higher(float(o["r2_w"]), float(m["r2_w"]))
    t_ot, t_mc = f"${fmt(float(o['per_iter_s']))}$", f"${fmt(float(m['per_iter_s']))}$"
    return rf"""\begin{{table}}[tb]
  \centering
  \caption{{\textbf{{CIS cell-interaction benchmark.}} Held-out distributional
    metrics (EMD/$W_1$, $W_2$, squared Bures--Wasserstein, MMD), functional
    recovery of the learned drift/interaction potentials (pattern-$R^2$ of
    $V_\theta$ vs.\ $V^\star$ and radial $W_\theta$ vs.\ $W^\star$), and training
    cost, OT vs.\ McCann initialization (seed 0). Distributional metrics lower is
    better, $R^2$ higher is better; better init in bold.}}
  \label{{tab:mccann-cis}}
  \setlength{{\tabcolsep}}{{5pt}}
  \adjustbox{{max width=\linewidth}}{{%
  \begin{{tabular}}{{l cccc cc c}}
    \toprule
    Init & EMD/$W_1\downarrow$ & $W_2\downarrow$ & $\mathrm{{Bd}}^2_{{W_2}}\downarrow$
      & MMD$\downarrow$ & $R^2(V)\uparrow$ & $R^2(W)\uparrow$ & s/iter \\
    \midrule
    OT     & {emd[0]} & {w2[0]} & {bw2[0]} & {mmd[0]} & {r2v[0]} & {r2w[0]} & {t_ot} \\
    McCann & {emd[1]} & {w2[1]} & {bw2[1]} & {mmd[1]} & {r2v[1]} & {r2w[1]} & {t_mc} \\
    \bottomrule
  \end{{tabular}}%
  }}
\end{{table}}"""


def _findings(ot: dict, mc: dict) -> str:
    """Data-derived findings: synthetic-gallery aggregates + CIS functional recovery."""
    keys = list(ot)
    def mean(d, c): return sum(float(d[k][c]) for k in keys) / len(keys)
    def wins(c):    return sum(1 for k in keys if float(ot[k][c]) < float(mc[k][c]))
    n = len(keys)
    synth = (
        rf"Across the full gallery ({n} cells), OT achieves lower held-out EMD in "
        rf"${wins('test_emd')}/{n}$ cells and lower $L^2$-UVP in ${wins('l2_uvp')}/{n}$; "
        rf"mean EMD ${mean(ot,'test_emd'):.2f}$ (OT) vs.\ ${mean(mc,'test_emd'):.2f}$ (McCann). "
    )
    try:
        co = _load(RESULTS / "cis" / "metrics" / "metrics.csv", lambda r: r["variant"])["cis"]
        cm = _load(RESULTS / "cis_mccann" / "metrics" / "metrics.csv", lambda r: r["variant"])["cis"]
        cis = (
            rf"On CIS the held-out distributions are comparable (McCann even attains a lower "
            rf"squared Bures--Wasserstein, ${fmt(float(cm['bw2']))}$ vs.\ ${fmt(float(co['bw2']))}$), "
            rf"yet its learned potentials are markedly worse: pattern-$R^2$ falls from "
            rf"${fmt(float(co['r2_v']))}/{fmt(float(co['r2_w']))}$ ($V/W$, OT) to "
            rf"${fmt(float(cm['r2_v']))}/{fmt(float(cm['r2_w']))}$ (McCann). "
        )
    except (FileNotFoundError, KeyError):
        cis = ""
    return synth + cis


SCHEMES = r"""\subsection{Initialization schemes}
\label{app:mccann-schemes}

\paragraph{Setup.} Stitching optimizes a particle bundle
$x_i(\tau_s)\in\mathbb{R}^D$, $i=1,\dots,N$, on the uniform grid
$\tau_s = t_0 + \tfrac{s-1}{T-1}(t_K-t_0)$, $s=1,\dots,T$, where $t_0<\dots<t_K$
are the observed times; write $a_s = (s-1)/(T-1)\in[0,1]$ for the normalized
grid. Both schemes seed the bundle from the two \emph{terminal} snapshots only,
and both are displacement (McCann) interpolations
\begin{equation}
  \mu_a \;=\; \bigl((1-a)\,\mathrm{id} + a\,T\bigr)_\#\,\mu_0 ,
  \qquad a\in[0,1],
  \label{eq:displacement}
\end{equation}
of a quadratic-cost optimal-transport map $T$ from $\mu_0$ to $\mu_1$; by
McCann's theorem \eqref{eq:displacement} is the constant-speed $W_2$ geodesic,
so $W_2(\mu_a,\mu_b)=|a-b|\,W_2(\mu_0,\mu_1)$. The schemes differ only in
\emph{which} pair of measures $T$ transports: the empirical terminal snapshots
(OT) or their Gaussian projections (McCann).

\paragraph{OT initialization (default).} Let
$\hat\mu_0 = \tfrac{1}{n_0}\sum_{i} \delta_{x^0_i}$ and
$\hat\mu_K = \tfrac{1}{n_K}\sum_{j} \delta_{x^K_j}$ be the empirical first and
last snapshots. For $n_0=n_K=n$ the Kantorovich problem with cost
$c(x,y)=\lVert x-y\rVert_2^2$ over doubly stochastic couplings attains its
optimum at a permutation matrix (Birkhoff), i.e.
\begin{equation}
  \sigma^\star \;=\; \operatorname*{arg\,min}_{\sigma\in S_n}\;
    \sum_{i=1}^{n} \bigl\lVert x^0_i - x^K_{\sigma(i)} \bigr\rVert_2^2 ,
\end{equation}
which we solve \emph{exactly} (no entropic regularization) with the Hungarian
algorithm, $O(n^3)$ time and $O(n_0 n_K)$ memory for the cost matrix. The bundle
is the straight-line interpolation of the matched pairs,
\begin{equation}
  x_i(a) \;=\; (1-a)\,x^0_{\pi(i)} + a\,x^K_{\sigma^\star(\pi(i))} + \xi_i ,
\end{equation}
where $\pi$ resamples $N$ of the $\min(n_0,n_K)$ matched pairs (with replacement
if $N$ exceeds that count) and $\xi_i\sim\mathcal{N}\!\bigl(0,\operatorname{diag}(0.01\,r)^2\bigr)$,
with $r$ the per-dimension data range, is a small jitter drawn once per particle
and \emph{held fixed across $s$}, so it displaces each path rigidly rather than
roughening it; it breaks degeneracy between duplicated pairs. Up to this jitter
and the resampling, $a\mapsto\operatorname{law}(x(a))$ is exactly the $W_2$
geodesic between $\hat\mu_0$ and $\hat\mu_K$, and particle identities are
inherited from real data rows.

\paragraph{McCann initialization.} Fit Gaussian moments
$(\hat m_0,\hat C_0)$ and $(\hat m_K,\hat C_K)$ to the same two snapshots and
regularize $C = \hat C + \varepsilon I$ with $\varepsilon=10^{-6}$ (the $t_0$
snapshot is often a tight, near-singular blob). Between nondegenerate Gaussians
the Brenier map is affine, $T(x) = m_K + A\,(x-m_0)$, with $A$ the unique
symmetric positive-definite solution of $A\,C_0\,A = C_K$:
\begin{equation}
  A \;=\; C_0^{-1/2}\bigl(C_0^{1/2} C_K C_0^{1/2}\bigr)^{1/2} C_0^{-1/2} .
  \label{eq:bures-map}
\end{equation}
All roots in \eqref{eq:bures-map} are the unique symmetric PSD square roots
$V\operatorname{diag}(\sqrt{w})V^{\!\top}$ obtained from a symmetric
eigendecomposition; this form is rotation-equivariant
($\sqrt{RCR^{\!\top}} = R\sqrt{C}R^{\!\top}$) and cancels eigenvector sign
ambiguity, hence continuous in the covariance --- a Cholesky factor would be
coordinate-order dependent and inject an axis-aligned shear. Drawing one batch
$\varepsilon_i\sim\mathcal{N}(0,I_D)$ that is \emph{shared across all grid
points}, \eqref{eq:displacement} gives
\begin{equation}
  x_i(0) = m_0 + C_0^{1/2}\varepsilon_i ,
  \qquad
  x_i(a) = m(a) + \bigl[(1-a)I + aA\bigr]\bigl(x_i(0)-m_0\bigr),
\end{equation}
with $m(a) = (1-a)m_0 + a\,m_K$. The induced law is
$\mathcal{N}\!\bigl(m(a),\,C_a\bigr)$ with
$C_a = [(1-a)I+aA]\,C_0\,[(1-a)I+aA]$ --- the Bures--Wasserstein geodesic,
matching both terminal moments exactly at $a\in\{0,1\}$. Since $(1-a)I+aA$ is
SPD for every $a\in[0,1]$, it is injective, so distinct $\varepsilon_i$ give
paths that never intersect: the bundle is laminar (non-braiding) by
construction rather than by luck, unlike seedings that interpolate covariance
factors or covariances directly, whose eigen-frame rotates with $a$ when $C_0$
and $C_K$ are anisotropic with different orientations. No jitter is added --- the
$\varepsilon_i$ already separate particles, and jitter would perturb the exact
terminal covariances. Cost is $O(D^3)$ (three eigendecompositions and one
inverse) plus $O(NTD^2)$ for the push-forward, independent of the snapshot
sizes.

\paragraph{Relation.} Both schemes produce a $W_2$ geodesic; OT takes it in the
full space of empirical measures, McCann in the Bures--Wasserstein submanifold
reached by projecting each terminal snapshot onto its first two moments. McCann
therefore discards everything beyond mean and covariance --- multimodality,
skew, curved supports --- and the two coincide in law (as $n\to\infty$) exactly
when the terminal snapshots are Gaussian. In exchange it avoids the $O(n^3)$
assignment and the $n\times n$ cost matrix, and guarantees a non-braiding
bundle. When only one time is observed ($K=0$) both degenerate to a constant
bundle: OT broadcasts resampled $t_0$ rows, McCann broadcasts
$m_0 + C_0^{1/2}\varepsilon_i$."""


def preamble(findings: str) -> str:
    return rf"""% !TeX root = (appendix fragment -- \input into the paper)
% Auto-generated by experiments/baselines/mccann_ablation_table.py from
% results/<exp>[_mccann]/metrics/*.csv. Requires
% \usepackage{{amsmath,amssymb,booktabs,multirow,adjustbox,graphicx,subcaption}}.

\section{{Trajectory-Initialization Ablation: OT vs.\ McCann}}
\label{{app:mccann-ablation}}

We ablate the scheme used to seed the Stitching particle bundle, holding data,
architecture, optimizer, and seed (0) fixed and varying only the initial
trajectory. \textbf{{OT}} (default) Hungarian-matches the first and last observed
snapshots and interpolates linearly; \textbf{{McCann}} interpolates along the
Bures--Wasserstein geodesic between Gaussian fits of the terminal snapshots. The
learned potential $V^\theta$ is trained to convergence in both cases. We report
the synthetic gallery (\autoref{{tab:mccann-synthetic}}), the EB scRNA-seq
interpolation benchmark in both the full-data (\autoref{{tab:mccann-eb-full}}) and
gappy leave-two-out (\autoref{{tab:mccann-eb}}) protocols, the CIS benchmark
(\autoref{{tab:mccann-cis}}), and the learned potentials and dynamics across all
benchmarks (Figures~\ref{{fig:mccann-recovery}}--\ref{{fig:mccann-gallery-unpaired}}).

{SCHEMES}

\paragraph{{Findings.}} {findings}On EB the two initializations are within noise
across all 5 seeds, with OT marginally ahead in both the full-data and gappy
leave-two-out protocols. McCann is competitive on near-isotropic synthetic
landscapes (e.g.\ Sphere, Bohachevsky) but degrades on the anisotropic ones,
consistent with it projecting each terminal snapshot onto its first two moments:
its Gaussian-geodesic seed departs from OT's empirical interpolation precisely
under anisotropic, differently-oriented terminal covariances, a regime these
distributional benchmarks do not emphasize but which the CIS potentials expose.
We retain OT as the default and report McCann as a sound alternative for marginal
reconstruction that can nonetheless miss the generating potentials (CIS).

\paragraph{{Reproducibility notes.}} The synthetic gallery and the CIS benchmark
are seed 0 (2000 epochs / benchmark defaults); the EB full-data and gappy tables
average 5 seeds (0--4), reported as mean${{\pm}}$std. Results live in
\texttt{{results/<exp>\_mccann/}} alongside the untouched OT baselines in
\texttt{{results/<exp>/}}.
"""


# Per-tree figure PDFs mirrored into docs/figures/ so the appendix is
# self-contained (dest filename -> source under results/). The
# potential_recovery_{paired,unpaired}.pdf pair is rendered separately by
# experiments/baselines/mccann_ablation_figure.py.
FIGURES_SRC = {
    "wavy_valley_init_grid.pdf": "wavy_valley_init/figures/wavy_valley_init_grid.pdf",
    "rna_panels_ot.pdf": "rna/figures/rna_panels.pdf",
    "rna_panels_mccann.pdf": "rna_mccann/figures/rna_panels.pdf",
    "cis_snapshots_ot.pdf": "cis/figures/cis_snapshots.pdf",
    "cis_snapshots_mccann.pdf": "cis_mccann/figures/cis_snapshots.pdf",
    "chiral_snapshots_ot.pdf": "chiral/figures/chiral_snapshots.pdf",
    "chiral_snapshots_mccann.pdf": "chiral_mccann/figures/chiral_snapshots.pdf",
    "synthetic_panel_paired_ot.pdf": "synthetic/figures/synthetic_panel_paired.pdf",
    "synthetic_panel_paired_mccann.pdf": "synthetic_mccann/figures/synthetic_panel_paired.pdf",
    "synthetic_panel_unpaired_ot.pdf": "synthetic/figures/synthetic_panel_unpaired.pdf",
    "synthetic_panel_unpaired_mccann.pdf": "synthetic_mccann/figures/synthetic_panel_unpaired.pdf",
}


def _copy_figures() -> None:
    """Mirror per-tree figure PDFs into docs/figures/ (warn on any missing source)."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for dest, src in FIGURES_SRC.items():
        s = RESULTS / src
        if s.exists():
            shutil.copyfile(s, FIGDIR / dest)
        else:
            print(f"  WARNING: missing figure source {src}")


def _pair_figure(label: str, caption: str, ot_pdf: str, mc_pdf: str) -> str:
    """A two-panel figure stacking the OT (top) and McCann (bottom) renderings."""
    return rf"""\begin{{figure}}[tb]
  \centering
  \begin{{subfigure}}{{\linewidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/{ot_pdf}}}
    \caption{{OT initialization (default).}}
  \end{{subfigure}}\\[0.6em]
  \begin{{subfigure}}{{\linewidth}}
    \centering
    \includegraphics[width=\linewidth]{{figures/{mc_pdf}}}
    \caption{{McCann initialization.}}
  \end{{subfigure}}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{figure}}"""


FIG_RECOVERY = r"""\begin{figure}[tb]
  \centering
  \begin{subfigure}{\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/potential_recovery_paired.pdf}
    \caption{Paired (matched) supervision.}
  \end{subfigure}\\[0.6em]
  \begin{subfigure}{\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/potential_recovery_unpaired.pdf}
    \caption{Unpaired supervision.}
  \end{subfigure}
  \caption{\textbf{Potential recovery on the 6 sensitive synthetic landscapes.}
    Rows within each panel: ground-truth $V^\star$ (top), potential learned from
    the OT initialization (middle), and from the McCann initialization (bottom);
    per-column pattern-$R^2$ annotated. Fields mean-centred; symmetric colour
    scale shared across rows within each column.}
  \label{fig:mccann-recovery}
\end{figure}"""

FIG_WAVY = r"""\begin{figure}[tb]
  \centering
  \includegraphics[width=\linewidth]{figures/wavy_valley_init_grid.pdf}
  \caption{\textbf{Wavy-valley landscape.} Ground-truth potential $V^\star$ and the
    potentials $V_\theta$ learned from the OT and McCann initializations, with the
    seeded and learned trajectory bundles overlaid (paper Fig.~1 analogue).}
  \label{fig:mccann-wavy}
\end{figure}"""


def figures() -> str:
    fig_eb = _pair_figure(
        "fig:mccann-eb-panels",
        r"\textbf{EB scRNA-seq learned potentials (Fig.~3 analogue).} Static "
        r"$V_\theta(x)$ (top row of each panel) and time-conditioned $V_\theta(x,t)$ "
        r"(bottom row) with held-out test cells and Stitching samples at observed and "
        r"interpolated timepoints, under the OT (top) and McCann (bottom) "
        r"initializations.",
        "rna_panels_ot.pdf",
        "rna_panels_mccann.pdf",
    )
    fig_cis = _pair_figure(
        "fig:mccann-cis-snap",
        r"\textbf{CIS cell-interaction snapshots (Fig.~4 analogue).} Data (top) and "
        r"model-sampled (bottom) snapshots across time, plus the learned $V_\theta$ "
        r"contour and radial $W_\theta$ against ground truth, under the OT (top) and "
        r"McCann (bottom) initializations.",
        "cis_snapshots_ot.pdf",
        "cis_snapshots_mccann.pdf",
    )
    fig_chiral = _pair_figure(
        "fig:mccann-chiral",
        r"\textbf{Chiral orbiting dynamics (Fig.~5 analogue).} Data and model-sampled "
        r"snapshots across time under the OT (top) and McCann (bottom) initializations.",
        "chiral_snapshots_ot.pdf",
        "chiral_snapshots_mccann.pdf",
    )
    fig_gallery_paired = _pair_figure(
        "fig:mccann-gallery-paired",
        r"\textbf{Full synthetic gallery, paired supervision (App.~F analogue).} "
        r"Ground-truth vs.\ learned potentials across all 15 landscapes, OT (top) vs.\ "
        r"McCann (bottom).",
        "synthetic_panel_paired_ot.pdf",
        "synthetic_panel_paired_mccann.pdf",
    )
    fig_gallery_unpaired = _pair_figure(
        "fig:mccann-gallery-unpaired",
        r"\textbf{Full synthetic gallery, unpaired supervision (App.~F analogue).} "
        r"Ground-truth vs.\ learned potentials across all 15 landscapes, OT (top) vs.\ "
        r"McCann (bottom).",
        "synthetic_panel_unpaired_ot.pdf",
        "synthetic_panel_unpaired_mccann.pdf",
    )
    return "\n\n".join(
        [FIG_RECOVERY, FIG_WAVY, fig_eb, fig_cis, fig_chiral, fig_gallery_paired, fig_gallery_unpaired]
    )


def main() -> None:
    _copy_figures()
    ot, mc = _synth()
    parts = [
        preamble(_findings(ot, mc)),
        table1(ot, mc),
        "",
        table_eb_full(),
        "",
        table_eb_gappy(),
        "",
        table_cis(),
        "",
        figures(),
    ]
    OUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

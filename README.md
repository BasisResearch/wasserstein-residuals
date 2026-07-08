# Stitching

> The method and its Python package are called **Stitching**; this is the
> `wasserstein-residuals` repository that hosts them.

Learning population dynamics from temporal snapshots via Wasserstein gradient
flows. `stitching/` is the importable library (models + harness);
`experiments/` holds one declarative **spec** per experiment, run through a single
**stage engine** (`python -m experiments`), that reproduces the paper.

Stitching represents the population as a KDE around a trainable particle
trajectory and fits it by penalising the **velocity residual** of the Wasserstein
gradient flow. Its Wasserstein gradient is a **closed form** —
`∇_W F(y) = c_H·∇log ρ(y) + c_V·∇V_θ(y) + c_W·Σ_j w_j ∇W_θ(y−x_j)` — so the package
is self-contained (the entropy term is the analytic KDE score; the potential and
interaction terms are plain `jax.grad`). See [stitching/velocity.py](stitching/velocity.py).

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

(or, without uv: `pip install -e .`)

## Layout

```
stitching/      installable library
  _kde/         KDE primitives: kernels, KernelDensity/Empirical, analytic kde_score, curves
  velocity.py   the closed-form Wasserstein gradient (WGFFunctional + wasserstein_grad)
  residual.py   WGF velocity residuals
  models/       stitching.py (the method) + lightspeed.py (JKOnet★ baseline)
  synthetic/    framework-free SDE simulator + analytic potentials
  config/build/train/evaluate/data + utils/  the experiment harness
configs/        data + method + experiment JSON presets (top-level; resolved from the
                repo, or `$STITCHING_CONFIG_DIR`)
data/           benchmark datasets (chiral, cis, rna committed under data/; the
                synthetic experiments simulate live from a potential — see data/README.md)
experiments/    one ExperimentSpec per experiment, run through a stage engine
  __main__.py   `python -m experiments <stage> <name>`  (data/train/eval/plot/all)
  _engine.py    the stage runner + Context (config build, checkpoint reload, idempotency)
  registry.py   EXPERIMENT_REGISTRY: name -> ExperimentSpec
  defs/         one <name>.py spec each: chiral, cis, doublewell, rna, synthetic, wavy_valley
  baselines/    external JKOnet★ / iJKOnet drivers (foreign venv; outside the engine)
results/        per-experiment outputs: results/<exp>/{runs,metrics,figures} (gitignored)
tests/          velocity-parity oracle, config-loader units, e2e regression gate
```

## Quick check

```bash
JAX_PLATFORMS=cpu uv run python -m experiments all wavy_valley --smoke
JAX_PLATFORMS=cpu uv run python -m experiments all synthetic --smoke
```

## Reproducing the paper

Every experiment runs through one entry point — `python -m experiments <stage> <name>`
— with decoupled stages (`data`/`train`/`eval`/`plot`/`all`) so re-rendering a figure
or recomputing a metric never retrains (`plot`/`eval` reload checkpoints). `--smoke`
gives a fast few-epoch pipeline check (drop it for the full run). Outputs land under
`results/<name>/{runs,metrics,figures}` and are gitignored (override the root with
`$STITCHING_RESULTS_DIR`). `python -m experiments list` shows everything; see
[experiments/README.md](experiments/README.md) for the engine details.

| Paper section | Figures / Tables | Experiment | Prereqs |
|---|---|---|---|
| **§5.1** Illustrative: continuous flow vs JKO chord | Fig. 1 | `wavy_valley` | — (synthetic; Stitching + Lightspeed) |
| **§5.2** Synthetic potential recovery | Table 1; Fig. 2; Table 4 / Figs. 6–8 | `synthetic` (+ §5.2 baselines below) | — (synthetic); Table 1 needs JKOnet★/iJKOnet |
| **§5.3** Single-cell trajectory inference (+ leave-two-out, Table 5) | Fig. 3, Tables 2, 5 | `rna` | `data/RNA_PCA_5/` (included) |
| **§5.4** Recovering interaction dynamics | Fig. 4, Table 3 | `cis` | `data/cis/` (included) |
| **§5.5** Recovering non-gradient flows | Fig. 5 | `chiral` | `data/chiral/` (included) |

The `rna`, `cis`, and `chiral` datasets are committed under `data/` (see
[data/README.md](data/README.md) for provenance and third-party attribution), so a
fresh clone runs end-to-end with no separate fetch step. One further experiment
ships beyond the paper tables: `doublewell` (an illustrative 3-regime panel). All
experiments train **Stitching**; `wavy_valley` additionally trains **Lightspeed**
(the in-repo JKOnet★). Run, e.g.:

```bash
uv run python -m experiments all  rna           # data → train → eval → plot (add --smoke for a quick check)
uv run python -m experiments plot rna           # re-render figures from saved checkpoints (no retrain)
```

### §5.2 synthetic potential recovery

Stitching (ours) is self-contained:

```bash
uv run python -m experiments all  synthetic                            # panels + Table 4 metrics
uv run python -m experiments plot synthetic --plot paired_vs_unpaired  # Fig. 2 / App. F Fig. 6
uv run python -m experiments plot synthetic --plot compact             # compact 3×6 true/paired/unpaired panel
```

The **Table 1** absolute-number comparison additionally needs the external
JKOnet★ and iJKOnet baselines (see [Baseline checkouts](#baseline-checkouts)):

```bash
export JKONET_STAR_DIR=…   IJKONET_DIR=…
uv run python experiments/baselines/synthetic_export_jkonet.py   # export datasets
uv run python experiments/baselines/synthetic_run_jkonet.py      # train JKOnet★  (JKOnet★ env)
uv run python experiments/baselines/synthetic_eval_jkonet.py     # eval JKOnet★
uv run python experiments/baselines/synthetic_run_ijkonet.py     # train+eval iJKOnet
uv run python experiments/baselines/synthetic_build_table.py     # → Table 1 LaTeX (stdout)
```

### §5.4 CIS JKOnet★ baseline (optional)

```bash
uv run python experiments/baselines/run_cis_jkonet.py export   # export CIS data
uv run python experiments/baselines/run_cis_jkonet.py train    # train JKOnet★ (JKOnet★ env)
uv run python experiments/baselines/run_cis_jkonet.py eval     # eval → cis/ metrics + figure
```

## Baseline checkouts

The synthetic and CIS comparisons against JKOnet★ and iJKOnet drive the upstream
codebases directly (no reimplementation). Clone them and point the scripts at the
checkouts:

```bash
git clone https://github.com/antonioterpin/jkonet-star.git
# … set up its venv, then:
export JKONET_STAR_DIR=$(realpath jkonet-star)

git clone https://github.com/iciemc/iJKOnet.git
# … set up its venv, then:
export IJKONET_DIR=$(realpath iJKOnet)
```

The wrapper scripts call the upstream `train.py` as subprocesses, parse metrics
from stdout, and forward-roll saved parameters through our evaluation pipeline at
the upstream default budgets (JKOnet★: 100 epochs; iJKOnet: 2000 epochs; both with
the `(64, 64)` MLP $V^\theta$).

## Test

```bash
make test      # fast suite (deselects the slow end-to-end parity gate)
make test-all  # everything, including the slow parity run
```

`tests/test_velocity_parity.py` cross-checks the closed-form Wasserstein gradient
against an independent autodiff oracle to a tight tolerance ("tested == shipped").
The slow gate's reference CSV is platform-baked, so `make test` deselects it.

## Contributors

- [Markus Heinonen](https://github.com/markusheinonen)
- Yair Shenfeld
- [Ricardo Baptista](https://github.com/baptistar)
- [Daniel Waxman](https://github.com/DanWaxman)
- [Dmitry Batenkov](https://github.com/dimkab)
- [Tim Cooijmans](https://github.com/cooijmanstim)
- [Eli Bingham](https://github.com/eb8680)
- [Fedor Sergeev](https://github.com/TheodorSergeev)

## Citation

If you use the code from this repository, please consider citing the paper:

```bibtex
@article{heinonen2026stitching,
  title   = {Stitching: Learning Population Dynamics via Wasserstein Gradient Flows},
  author  = {Heinonen, Markus and Shenfeld, Yair and Baptista, Ricardo and Waxman, Daniel and Batenkov, Dmitry and Cooijmans, Tim and Bingham, Eli and Sergeev, Fedor},
  journal = {arXiv preprint arXiv:2607.04738},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.04738}
}
```

## Provenance & licence

Apache-2.0 (see [LICENSE.md](LICENSE.md)). The KDE primitives under `stitching/_kde/`
are a self-contained, autodiff-free core: the entropy term of the Wasserstein
gradient is the hand-derived closed-form KDE score (the potential and interaction
terms are plain `jax.grad` of the networks), so there is no generic
Wasserstein-autodiff dependency — no differentiating through an OT solve.

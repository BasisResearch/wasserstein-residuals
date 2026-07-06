# experiments

Every paper experiment is a declarative **spec** run through one **stage engine**.
Running any experiment is one line; adding one is a single file.

```bash
python -m experiments list                 # what's available
python -m experiments all   chiral         # data → train → eval → plot
python -m experiments plot  chiral         # re-render from saved checkpoints (no retrain)
python -m experiments all   chiral --smoke # fast, isolated to results/chiral_smoke/
```

## Stages (the checkpoint is the seam)

`save_run`/`load_run` already emit a self-contained checkpoint (`model.eqx` +
`cfg.json` + `data.npz` + `manifest.json`). The engine promotes that into
uniform, **decoupled** stages so changing a plot or a metric never retrains:

| Stage | Reads | Writes | Reruns when |
| --- | --- | --- | --- |
| `data` | preset | — (reports shapes; inspect/prefetch) | — |
| `train` | data | `results/<exp>/runs/<variant>/s<seed>/` checkpoint | a model/variant/data/commit changes |
| `eval` | checkpoints | `results/<exp>/metrics/*.csv` (+ per-run metrics) | a metric changes |
| `plot` | checkpoints + CSV | `results/<exp>/figures/*.pdf` | a figure changes |
| `all` | — | train → eval → plot | — |

One experiment's whole output is the single tree `results/<exp>/{runs,metrics,figures}`:
independent per-model checkpoints under `runs/`, with `eval`/`plot` the collection
stages that gather them into `metrics/` CSVs and `figures/` (vector PDF). There is
no separate paper-figure location — point the paper's `\includegraphics` at
`results/<exp>/figures/`.

`train` is **idempotent**: a checkpoint whose manifest fit-id matches the current
`(commit + data)` is reused, so a multi-model figure retrains only what changed.
Pass `--force`, or name a `--variant`, to redo just that. (Uncommitted code edits
share a commit+data fit-id, so use `--force` while iterating on a model.)

`all` runs `train → eval → plot` (not `data` — `train` already materialises each
cell's `data.npz`; the checkpoint is the authoritative dataset snapshot). `data`
is a standalone inspect/prefetch stage.

### Options

`--smoke` (fast, wall-time-only, isolated `<exp>_smoke/` dir) · `--force`
(retrain fresh checkpoints) · `--variant V` / `--seed N` (repeatable; restrict
the sweep) · `--plot P` (repeatable; render only named plots) · `--out-dir DIR`.

## Layout

```
experiments/
  __main__.py     # `python -m experiments <stage> <name>` → engine
  _engine.py      # the stage runner + Context (cfg build, checkpoint reload, idempotency)
  registry.py     # EXPERIMENT_REGISTRY: name -> ExperimentSpec
  defs/
    <name>.py     # one ExperimentSpec per experiment: variants + eval + plots
  baselines/      # foreign-venv (jkonet-star) glue — NOT part of this engine
```

The `baselines/` tree is **not** run by this engine. It is jkonet-star / iJKOnet,
a different method family. The heavy lifting — the upstream `data_generator.py` /
`train.py` — runs as subprocesses in jkonet-star's own virtualenv
(`../jkonet-star/.venv`), which **cannot import `stitching`** (conflicting deps).
The stitching-side glue here (e.g. `synthetic_export_jkonet.py`) runs in *this*
repo's env: it exports datasets into the upstream format and conforms the result
CSVs to the leaderboard schema. Each script carries its own usage in its module
docstring.

## Adding an experiment

Drop `defs/<name>.py` and list it in `defs/__init__.py`:

```python
from experiments.registry import ExperimentSpec, Variant, register

def _plot_main(ctx):
    run = ctx.load("main")          # reload the checkpoint
    ...                              # fig.savefig(ctx.figure_path("figure"))  # PDF under figures/

def _evaluate(ctx):                  # optional; omit for wall-time-only runs
    rows = [...]                     # reload checkpoints via ctx.load_all(), compute metrics
    ctx.write_csv("metrics.csv", rows, fieldnames=(...))

register(ExperimentSpec(
    name="<name>",
    description="one line",
    variants=(Variant("main", "<data>:<method>:<experiment>"),),
    seeds=(0,),                      # crossed with variants
    evaluate=_evaluate,
    plots={"main": _plot_main},
))
```

A `Variant` is `(name, composite-preset, overrides)`; the sweep is
`variants × seeds`. Config presets live in the top-level `configs/` tree
(`configs/{data,method,experiment}/`). The genuinely experiment-specific code is
the `evaluate`/`plots` hooks; everything else (CLI, output dirs, idempotent
training, checkpoint reload) is the engine.

## Tests

`tests/test_experiments_registry.py` checks every spec resolves its configs and
declares callable hooks. `tests/test_parity.py` runs the `synthetic` experiment
under `--smoke` and pins its metrics CSV bitwise.

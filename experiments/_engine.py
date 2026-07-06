"""The stage engine: runs an :class:`~experiments.registry.ExperimentSpec`.

The checkpoint *is* the stage seam. ``save_run`` / ``load_run`` already emit a
self-contained run dir (``model.eqx`` + ``cfg.json`` + ``data.npz`` +
``manifest.json``); this engine promotes that into explicit, uniform stages every
experiment shares:

    data   load each cell's dataset and report (inspect/prefetch; the
           authoritative dataset snapshot is each checkpoint's ``data.npz``).
    train  fit every (variant, seed) cell → a checkpoint under ``runs/``.
           Idempotent: a checkpoint whose manifest fit-id matches the current
           (commit + data) is reused; ``--force`` or a named ``--variant``
           retrains. So a multi-model figure retrains only what changed.
    eval   reload checkpoints, compute metrics, write the experiment's CSV(s)
           under ``metrics/``. NO retraining.
    plot   reload checkpoints (and/or read a ``metrics/`` CSV), render figures as
           vector PDFs under ``figures/``. NO retraining, NO re-eval.
    all    train → eval → plot (``train`` materialises each cell's ``data.npz``,
           so a separate ``data`` pass is redundant here).

    So one experiment's whole output tree is ``results/<exp>/{runs,metrics,figures}``:
    independent per-model checkpoints under ``runs/``, with ``eval``/``plot`` as the
    collection stages that gather them into ``metrics/`` CSVs and ``figures/`` PDFs.

``Context`` carries the per-invocation state (output dir, smoke flag) and the
helpers a spec's ``evaluate`` / ``plots`` hooks use to rebuild configs and reload
checkpoints. Heavy deps (jax, the model/data stack) are imported lazily inside
the stage bodies so importing this module — or just *listing* experiments — stays
fast.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.registry import ExperimentSpec, Variant, get_experiment
from stitching.utils.paths import results_dir
from stitching.utils.runners import repo_root

if TYPE_CHECKING:
    from stitching.config import Config
    from stitching.utils.persistence import Run

STAGES = ("data", "train", "eval", "plot", "all")


class Context:
    """Per-invocation engine state plus the helpers a spec's hooks call.

    A spec's ``evaluate`` / ``plots`` hooks receive a ``Context`` and use it to
    rebuild a cell's :class:`~stitching.config.Config` (:meth:`build_cfg`), reload
    checkpoints (:meth:`load` / :meth:`load_variants` / :meth:`load_all`), read
    the metrics CSV (:meth:`read_csv`) and write outputs (:meth:`write_csv` →
    ``metrics/``, :meth:`figure_path` → ``figures/``).
    """

    def __init__(self, spec: ExperimentSpec, out_dir: Path, *, smoke: bool) -> None:
        """Bind a spec to a resolved output dir and the smoke flag.

        Args:
            spec: The experiment being run.
            out_dir: The resolved ``results/<name>[_smoke]`` (or ``--out-dir``).
            smoke: Whether this is a fast, wall-time-only smoke run.
        """
        self.spec = spec
        self.out_dir = out_dir
        self.smoke = smoke

    # --- paths --------------------------------------------------------------
    @property
    def runs_dir(self) -> Path:
        """The per-experiment checkpoint root, ``<out_dir>/runs``."""
        return self.out_dir / "runs"

    @property
    def figures_dir(self) -> Path:
        """The per-experiment figure tree, ``<out_dir>/figures`` (not created)."""
        return self.out_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        """The per-experiment metrics tree, ``<out_dir>/metrics`` (not created)."""
        return self.out_dir / "metrics"

    def figure_path(self, stem: str) -> Path:
        """Return (and ensure) ``<out_dir>/figures/<stem>.pdf`` for a plot to save.

        Figures are vector PDFs under the single ``results/<exp>`` tree — there is
        no separate paper-figure location; point the paper's ``\\includegraphics``
        at this path.
        """
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        return self.figures_dir / f"{stem}.pdf"

    def run_dir(self, variant_name: str, seed: int) -> Path:
        """Checkpoint dir for one cell: ``<out_dir>/runs/<variant>/s<seed>``."""
        return self.runs_dir / variant_name / f"s{seed}"

    # --- cells & configs ----------------------------------------------------
    def cells(
        self,
        variant_filter: Sequence[str] | None = None,
        seed_filter: Sequence[int] | None = None,
    ) -> list[tuple[Variant, int]]:
        """Return the ``(variant, seed)`` cells to act on, after CLI filtering.

        Honours the spec's ``--smoke`` variant subset, then any explicit
        ``--variant`` / ``--seed`` selection. Raises ``ValueError`` naming a
        requested variant/seed that does not exist (a typo fails loudly, not
        silently to an empty sweep).
        """
        variants = self.spec.selected_variants(smoke=self.smoke)
        if variant_filter:
            known = {v.name for v in self.spec.variants}
            unknown = [n for n in variant_filter if n not in known]
            if unknown:
                raise ValueError(
                    f"unknown --variant {unknown} for {self.spec.name!r}; "
                    f"choose from {sorted(known)}."
                )
            keep = set(variant_filter)
            variants = [v for v in variants if v.name in keep]
        seeds = list(self.spec.seeds)
        if seed_filter:
            unknown_s = [s for s in seed_filter if s not in set(seeds)]
            if unknown_s:
                raise ValueError(
                    f"unknown --seed {unknown_s} for {self.spec.name!r}; "
                    f"this experiment sweeps seeds {seeds}."
                )
            keep_s = set(seed_filter)
            seeds = [s for s in seeds if s in keep_s]
        return [(v, s) for v in variants for s in seeds]

    def build_cfg(self, variant: Variant, seed: int) -> Config:
        """Resolve a cell's config: preset → per-cell overrides → smoke trim.

        Mirrors what each old runner did by hand: ``load_config`` the composite,
        overlay the variant's sweep overrides (and the cell seed) via
        :func:`dataclasses.replace`, then apply the wall-time-only ``--smoke``
        trim. Smoke fields (epochs/num_particles/…) and override fields (the sweep
        axes) are disjoint in every experiment, so the result is order-independent
        and byte-identical to the pre-engine config.
        """
        from stitching.config import load_config
        from stitching.utils.runners import apply_smoke

        cfg = load_config(variant.config)
        overrides = dict(variant.overrides)
        if seed != cfg.seed:
            overrides["seed"] = seed
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        if self.smoke:
            cfg = apply_smoke(cfg, self.spec.smoke_epochs, **self.spec.smoke_overrides)
        return cfg

    # --- checkpoint reload --------------------------------------------------
    def load(
        self, variant_name: str, seed: int | None = None, *, strict: bool = True
    ) -> Run:
        """Reload a single cell's checkpoint as a :class:`~stitching.utils.persistence.Run`.

        Args:
            variant_name: Which variant to reload.
            seed: Which seed (defaults to the spec's first seed).
            strict: Forwarded to ``load_run`` (data-drift raises on the publish path).
        """
        return self.load_variants(variant_name, seed=seed, strict=strict)[variant_name]

    def load_variants(
        self, *names: str, seed: int | None = None, strict: bool = True
    ) -> dict[str, Run]:
        """Reload several cells (all at one seed) keyed by variant name.

        Uses ``load_runs`` so a missing/partial checkpoint fails loudly naming the
        variant (telling the user to run ``train`` first).
        """
        from stitching.utils.persistence import load_runs

        seed = self.spec.seeds[0] if seed is None else seed
        dirs = {n: self.run_dir(n, seed) for n in names}
        return load_runs(dirs, strict=strict)

    def load_all(self, *, strict: bool = True) -> dict[tuple[str, int], Run]:
        """Reload every cell, keyed by ``(variant_name, seed)`` (for eval sweeps)."""
        from stitching.utils.persistence import load_runs

        cells = self.cells()
        dirs = {f"{v.name}@s{s}": self.run_dir(v.name, s) for v, s in cells}
        loaded = load_runs(dirs, strict=strict)
        return {(v.name, s): loaded[f"{v.name}@s{s}"] for v, s in cells}

    # --- CSV io -------------------------------------------------------------
    def write_csv(
        self,
        filename: str,
        rows: Sequence[Mapping[str, Any]],
        fieldnames: Sequence[str],
    ) -> Path:
        """Write ``<out_dir>/metrics/<filename>`` as CSV; return the path."""
        from stitching.utils.runners import write_csv as _write_csv

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        return _write_csv(self.metrics_dir / filename, rows, fieldnames)

    def read_csv(self, filename: str) -> list[dict[str, str]]:
        """Read ``<out_dir>/metrics/<filename>`` as a list of row dicts.

        Raises ``FileNotFoundError`` pointing at the ``eval`` stage if the CSV is
        absent (the ``plot`` stage reads metrics the ``eval`` stage produced).
        """
        path = self.metrics_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run `eval` (or `all`) before `plot`."
            )
        with path.open(newline="") as f:
            return list(csv.DictReader(f))

    # --- idempotency --------------------------------------------------------
    def is_fresh(self, run_dir: Path, train_data: Any, test_data: Any) -> bool:
        """Whether *run_dir* holds a complete checkpoint for the current fit-id.

        Fresh ⇔ all run files present and the manifest's fit-id equals the
        ``(commit + data)`` fit-id of *train_data*/*test_data*. A different commit
        or byte-different data is stale; uncommitted code edits are not detected
        (same commit + data) — use ``--force`` while iterating on a model.
        """
        from stitching.utils.persistence import fit_id_for

        required = ("model.eqx", "cfg.json", "data.npz", "manifest.json")
        if any(not (run_dir / name).exists() for name in required):
            return False
        try:
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return False
        saved = manifest.get("fit_id")
        return bool(saved) and saved == fit_id_for(train_data, test_data)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _stage_data(
    ctx: Context,
    variant_filter: Sequence[str] | None,
    seed_filter: Sequence[int] | None,
) -> None:
    """Load each selected cell's dataset and report its shape (inspect/prefetch).

    Does not write a separate dataset cache: the authoritative per-cell snapshot
    is the ``data.npz`` that ``train`` writes into each checkpoint.
    """
    from stitching.data import load_data

    for variant, seed in ctx.cells(variant_filter, seed_filter):
        cfg = ctx.build_cfg(variant, seed)
        train_data, test_data = load_data(cfg)
        print(
            f"  {variant.name} s{seed}: train {tuple(train_data.x.shape)} "
            f"test {tuple(test_data.x.shape)}"
        )


def _stage_train(
    ctx: Context,
    variant_filter: Sequence[str] | None,
    seed_filter: Sequence[int] | None,
    *,
    force: bool,
) -> None:
    """Fit every selected cell into its checkpoint, skipping fresh ones."""
    from stitching.data import load_data
    from stitching.utils.persistence import save_run
    from stitching.utils.runners import fit_seeded

    for variant, seed in ctx.cells(variant_filter, seed_filter):
        cfg = ctx.build_cfg(variant, seed)
        train_data, test_data = load_data(cfg)
        run_dir = ctx.run_dir(variant.name, seed)
        if not force and ctx.is_fresh(run_dir, train_data, test_data):
            print(
                f"  {variant.name} s{seed}: fresh checkpoint — skip (--force to retrain)"
            )
            continue
        t0 = time.time()
        model = fit_seeded(cfg, train_data)
        wall = time.time() - t0
        save_run(
            run_dir, cfg, model, train_data, test_data, metrics={"wall_time_s": wall}
        )
        print(f"  {variant.name} s{seed}: trained in {wall:.0f}s → {run_dir}")


def _stage_eval(ctx: Context) -> None:
    """Run the experiment's metrics hook over reloaded checkpoints, if any."""
    if ctx.spec.evaluate is None:
        print(f"  {ctx.spec.name}: no eval stage (wall-time-only experiment)")
        return
    ctx.spec.evaluate(ctx)


def _stage_plot(ctx: Context, only: Sequence[str] | None) -> None:
    """Render the experiment's figures from reloaded checkpoints / metrics CSV."""
    plots = ctx.spec.plots
    if not plots:
        print(f"  {ctx.spec.name}: no plots declared")
        return
    if only:
        unknown = [n for n in only if n not in plots]
        if unknown:
            raise ValueError(
                f"unknown --plot {unknown} for {ctx.spec.name!r}; "
                f"choose from {sorted(plots)}."
            )
    for name, fn in plots.items():
        if only and name not in set(only):
            continue
        print(f"  plot: {name}")
        fn(ctx)


def run_experiment(
    stage: str,
    name: str,
    *,
    smoke: bool = False,
    force: bool = False,
    variant_filter: Sequence[str] | None = None,
    seed_filter: Sequence[int] | None = None,
    only_plots: Sequence[str] | None = None,
    out_dir: Path | None = None,
) -> None:
    """Run one *stage* of experiment *name* (the engine entry point).

    Args:
        stage: One of :data:`STAGES`.
        name: A registered experiment name.
        smoke: Fast wall-time-only run, isolated to a ``<name>_smoke`` dir.
        force: Retrain even when a fresh checkpoint exists (``train``/``all``).
        variant_filter: Restrict to these variant names.
        seed_filter: Restrict to these seeds.
        only_plots: For the ``plot`` stage, render only these named plots.
        out_dir: Explicit output dir (overrides the ``results/<name>`` default).
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {list(STAGES)}.")
    spec = get_experiment(name)

    if out_dir is not None:
        resolved = Path(out_dir)
        resolved.mkdir(parents=True, exist_ok=True)
    else:
        suffix = "_smoke" if smoke else ""
        resolved = results_dir(f"{spec.name}{suffix}", repo_root())
    ctx = Context(spec, resolved, smoke=smoke)

    print(f"[{stage}] {spec.name}{' (smoke)' if smoke else ''} → {resolved}")
    if stage == "data":
        _stage_data(ctx, variant_filter, seed_filter)
    elif stage == "train":
        _stage_train(ctx, variant_filter, seed_filter, force=force)
    elif stage == "eval":
        _stage_eval(ctx)
    elif stage == "plot":
        _stage_plot(ctx, only_plots)
    elif stage == "all":
        _stage_train(ctx, variant_filter, seed_filter, force=force)
        _stage_eval(ctx)
        _stage_plot(ctx, only_plots)

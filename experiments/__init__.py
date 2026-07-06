"""Experiment drivers, organised as declarative specs over a shared stage engine.

Every paper experiment is one :class:`~experiments.registry.ExperimentSpec` under
:mod:`experiments.defs`, registered in ``EXPERIMENT_REGISTRY``. A single engine
(:mod:`experiments._engine`) runs each spec through uniform, decoupled stages —
``data`` → ``train`` → ``eval`` → ``plot`` (or ``all``) — so changing a plot or a
metric never retrains, and ``train`` is idempotent (a fresh checkpoint is reused
unless the data or commit changed, or ``--force`` is given).

Run any experiment in one line::

    python -m experiments all <name>          # data → train → eval → plot
    python -m experiments plot <name>         # re-render from saved checkpoints
    python -m experiments list                # what's available

Add one in one place: a new ``experiments/defs/<name>.py`` declaring an
``ExperimentSpec`` and registering it. See ``experiments/README.md``.

This is a repo-local package (not installed); ``python -m experiments`` finds it
via the repo root on ``sys.path``. The ``baselines/`` subtree is foreign-venv
glue and is intentionally *not* part of this engine.
"""

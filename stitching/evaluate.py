"""Evaluation orchestration: sample the model, compute metrics, print.

The actual metric definitions (Wasserstein, MMD, Bures-Wasserstein, the
iJKOnet UVPs, functional accuracy, one-step-ahead) live in
:mod:`stitching.utils.metrics`. The model sampling lives on each
``Model.sample`` in :mod:`stitching.models`. This file is just the
orchestration that wires those together — sample once, compute the
metric battery, print results.
"""

from __future__ import annotations

import warnings
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from stitching._kde import SpatioTemporalData

from .build import spec_for_model
from .utils.metrics import (
    bures_wasserstein_sq,
    bw2_uvp,
    functional_accuracy,
    l2_uvp,
    mmd_dmsb,
    mmd_rbf,
    one_step_ahead,
    wasserstein,
)

METRIC_NAMES: tuple[str, ...] = ("emd", "w2", "w2_squared", "bw2", "mmd", "mmd_dmsb")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample(
    model,
    t_eval: jax.Array,
    *,
    train_data: SpatioTemporalData,
    num_samples: int | None,
    key: jax.Array,
) -> dict[float, jax.Array]:
    """One uniform sample call for both model classes.

    Each model does its own KDE-resampling (Stitching) / JKO chain
    (Lightspeed) inside ``Model.sample`` — this just routes the
    model-specific extras (`train_data`) via the registry capability flag.
    """
    if spec_for_model(model).needs_train_data_for_sample:
        return model.sample(t_eval, train_data)
    n = num_samples or train_data.x.shape[0]
    return model.sample(t_eval, num_samples=n, key=key)


# ---------------------------------------------------------------------------
# Test metrics: sample then compute
# ---------------------------------------------------------------------------


def evaluate_test_metrics(
    model,
    train_data: SpatioTemporalData,
    test_data: SpatioTemporalData,
    eval_num_samples: int | None = None,
    eval_reps: int = 1,
) -> dict[str, dict[str, float | dict[float, float]]]:
    """Sample the model and compute the full distributional-metric battery.

    Always computes EMD ($W_1$), $W_2$, $W_2^2$, squared Bures–Wasserstein,
    and multi-bandwidth Gaussian RBF MMD². For Stitching the metrics are
    averaged over ``eval_reps`` independent KDE-sampled clouds.

    Returns a dict keyed by metric name; each entry has ``"mean"``
    (averaged across timepoints) and ``"by_time"`` (per-timepoint dict).
    """
    t_unique = jnp.unique(test_data.t)
    test_targets: dict[float, jax.Array] = {}
    for t_val in t_unique:
        t_key = round(float(t_val), 8)
        mask = jnp.isclose(test_data.t, t_val, atol=1e-6)
        x_t = test_data.x[mask]
        if x_t.shape[0] > 0:
            test_targets[t_key] = x_t

    # Lightspeed is deterministic for a given key; we still vary the key per
    # rep so that KDE-resampling Stitching produces independent clouds.
    accum: dict[str, dict[float, list[float]]] = {k: {} for k in METRIC_NAMES}
    base_key = jax.random.key(42)

    for r in range(eval_reps):
        rep_key = jax.random.fold_in(base_key, r)
        samples = _sample(
            model,
            t_unique,
            train_data=train_data,
            num_samples=eval_num_samples,
            key=rep_key,
        )

        for t_key, target_x in test_targets.items():
            generated = samples.get(t_key)
            if generated is None:
                continue
            n = min(generated.shape[0], target_x.shape[0])
            a = np.asarray(generated[:n], dtype=np.float64)
            b = np.asarray(target_x[:n], dtype=np.float64)
            for name, val in (
                ("emd", wasserstein(a, b, metric="emd")),
                ("w2", wasserstein(a, b, metric="w2")),
                ("w2_squared", wasserstein(a, b, metric="w2_squared")),
                (
                    "bw2",
                    bures_wasserstein_sq(a, b) if a.shape[0] >= 2 else float("nan"),
                ),
                ("mmd", mmd_rbf(a, b)),
                ("mmd_dmsb", mmd_dmsb(a, b)),
            ):
                accum[name].setdefault(t_key, []).append(float(val))

    out: dict[str, dict[str, float | dict[float, float]]] = {}
    for name in METRIC_NAMES:
        by_time = {t: float(np.mean(vs)) for t, vs in accum[name].items()}
        mean = float(np.mean(list(by_time.values()))) if by_time else float("nan")
        out[name] = {"mean": mean, "by_time": by_time}
    return out


# ---------------------------------------------------------------------------
# Run-level orchestration
# ---------------------------------------------------------------------------


def run_blackbox_evaluation(
    model,
    cfg,
    train_data: SpatioTemporalData,
    test_data: SpatioTemporalData,
) -> tuple[float, dict[float, float], dict[str, dict], dict[str, dict]]:
    """Distributional metrics + (for Stitching) JKO one-step-ahead.

    Prints summaries and returns ``(headline_mean, headline_by_time,
    all_metrics, one_step_metrics)``.
    """
    metric: Literal["emd", "w2", "w2_squared"] = cfg.metric

    full = evaluate_test_metrics(
        model,
        train_data,
        test_data,
        eval_num_samples=cfg.eval_num_samples,
        eval_reps=cfg.eval_reps,
    )
    headline = full[metric]
    test_emd = float(headline["mean"])  # type: ignore[arg-type]
    emd_by_time = dict(headline["by_time"])  # type: ignore[arg-type]

    label = metric.upper()
    print(f"\nTest {label}:  mean={test_emd:.4f}", end="")
    if emd_by_time:
        t_min, v_min = min(emd_by_time.items(), key=lambda kv: kv[1])
        t_max, v_max = max(emd_by_time.items(), key=lambda kv: kv[1])
        print(f"   min={v_min:.4f} (t={t_min:g})   max={v_max:.4f} (t={t_max:g})")
    else:
        print()

    one_step: dict[str, dict] = {}
    if spec_for_model(model).supports_one_step_ahead:
        functional = model.functional()
        bw = model.bandwidth
        for one_metric in ("emd", "w2"):
            try:
                m, by_t = one_step_ahead(functional, bw, test_data, metric=one_metric)
                one_step[one_metric] = {
                    "mean": float(m),
                    "by_time": {float(t): float(v) for t, v in by_t.items()},
                }
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"one_step_ahead({one_metric!r}) failed and was skipped: "
                    f"{type(exc).__name__}: {exc}",
                    stacklevel=2,
                )

    return test_emd, emd_by_time, full, one_step


def run_synthetic_evaluation(
    model,
    cfg,
    train_data: SpatioTemporalData,
    test_data: SpatioTemporalData,
) -> tuple[
    float,
    dict[float, float],
    dict[str, float | dict[float, float]],
    dict[str, dict],
    dict[str, dict],
]:
    """Blackbox metrics + ground-truth-aware extras (functional accuracy, UVPs)."""
    from .synthetic import get_true_potential, get_true_potential_grad

    test_emd, emd_by_time, all_metrics, one_step_metrics = run_blackbox_evaluation(
        model, cfg, train_data, test_data
    )

    func: dict[str, float | dict[float, float]] = {}
    true_V = get_true_potential(cfg.potential)
    if true_V is not None and hasattr(model, "potential_net"):
        c_pot = float(getattr(model, "potential_coeff", 1.0))
        potential_fn = model.learned_potential_fn()
        learned_V = lambda pts: (
            c_pot
            * np.asarray(jax.vmap(potential_fn)(jnp.asarray(pts, dtype=jnp.float32)))
        )
        func.update(functional_accuracy(learned_V, true_V, train_data))
        true_grad = get_true_potential_grad(cfg.potential)
        if true_grad is not None:
            learned_grad = jax.vmap(jax.grad(lambda x: potential_fn(x).reshape(())))
            learned_grad_np = lambda pts: np.asarray(
                learned_grad(jnp.asarray(pts, dtype=jnp.float32))
            )
            l2_mean, l2_by_t = l2_uvp(
                learned_grad_np,
                true_grad,
                train_data,
                tau=float(cfg.synth_dt),
            )
            func["l2_uvp"] = l2_mean
            func["l2_uvp_by_time"] = l2_by_t
        _f = lambda k: float(func[k])  # type: ignore[arg-type]
        print("\nFunctional accuracy:")
        print(f"  Potential R²:   {_f('potential_r2'):.4f}")
        print(f"  Potential RMSE: {_f('potential_rmse'):.4f}")
        print(f"  Pattern R²:     {_f('pattern_r2'):.4f}  (scale-invariant)")
        print(f"  Pattern corr:   {_f('pattern_corr'):+.4f}")
        if "l2_uvp" in func:
            print(f"  L²-UVP:         {_f('l2_uvp'):.4f}%  (iJKOnet, ↓)")

    if spec_for_model(model).supports_bw2_uvp:
        t_unique = jnp.array(np.unique(np.asarray(test_data.t)))
        pred_samples = model.sample(t_unique, num_samples=None, key=None)
        pred_w = np.asarray(model.particle_weights())
        bw_mean, bw_by_t = bw2_uvp(pred_samples, test_data, pred_weights=pred_w)
        func["bw2_uvp"] = bw_mean
        func["bw2_uvp_by_time"] = bw_by_t
        print(f"  Bd²W₂-UVP:      {bw_mean:.4f}%  (iJKOnet, ↓)")

    return test_emd, emd_by_time, func, all_metrics, one_step_metrics

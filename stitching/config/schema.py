"""The :class:`Config` dataclass — the typed schema for a WGF benchmark run.

A single flat dataclass holds every knob (data, training, functional
coefficients, the Stitching/Lightspeed method hyperparameters, synthetic-data
generation, evaluation). Fields are grouped by concern below. ``from_dict``
validates (unknown keys are a loud error) and warns about method-field
mismatches.
"""

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(unsafe_hash=True)
class Config:
    """Typed configuration for a JAX WGF benchmark run."""

    # Dataset / split
    dataset: str | None = (
        None  # dataset name; see stitching.data.DATASET_REGISTRY (list_datasets())
    )
    potential: str | None = None  # synthetic system name (see list_systems())
    model: str = "stitching"  # model type: "stitching" or "lightspeed"
    seed: int = 0  # global random seed
    dims: int = 5  # PCA dimensionality for RNA data
    scale_time: bool = False  # whether to rescale observation times to [0, 1]
    split_seed: int = 0  # random seed for train/test split
    train_fraction: float = 0.6  # fraction of data used for training
    data_root: str | None = None  # override path to raw data directory
    split_path: str | None = None  # path to a precomputed train/test split .npz file

    # Training
    epochs: int = 100  # number of training epochs
    batch_size: int = 256  # mini-batch size
    lr: float = 1e-3  # learning rate (Adam)
    potential_hidden: tuple[int, ...] = (
        64,
        64,
        64,
    )  # hidden widths for potential / density / score networks

    # Functional coefficients
    entropy_coeff: float = 0.03  # weight on the entropy functional E[log ρ]
    potential_coeff: float = 1.0  # weight on the potential functional E[V(x)]
    interaction_coeff: float = 0.0  # weight on the interaction functional (0 = off)
    interaction_type: str = (
        "radial"  # interaction kernel: "radial" (MLP(||r||²)) or "full" (MLP(r))
    )
    interaction_hidden: tuple[int, ...] = (64, 64)  # hidden dims for interaction MLP

    # Loss weights
    w_nll: float = 1.0  # negative log-likelihood weight

    # Particle ODE / Stitching hyperparameters
    num_particles: int = 50  # number of learnable particles
    num_steps: int = 20  # number of ODE time-steps
    lengthscale: float | str = (
        "silverman"  # bandwidth: scalar = fixed everywhere, "silverman" = auto Silverman from t=0 particles
    )
    stitching_coeff: float = 1.0  # stitching consistency loss weight
    stitching_scheme: str = (
        "trapez"  # stitching discretisation: "trapez", "euler", "implicit", "midpoint"
    )
    stitching_norm: str = (
        "raw"  # residual normalisation: "raw", "relative", "relative-sg"
    )
    stitching_kinetic: float = 1.0  # kinetic coefficient λ for WGF force scaling
    stitching_residual_alpha: float = 0.0  # tradeoff in [0, 1] between centers quadrature (α=0, deterministic, O(σ²) bias) and KDE-MC with Nadaraya–Watson velocity (α=1, unbiased on the KDE, O(1/M) variance). Loss is (1-α)·L_centers + α·L_kde. Extremes short-circuit (no unused computation); 0 < α < 1 evaluates both estimators and convexly combines them. Both → continuum velocity residual as σ→0, N→∞.
    freeze_particle_weights: bool = True  # default: hold raw_weights at zero (uniform mixture). Trainable weights let surplus particles drift (low effective sample size), polluting the WGF residual signal that shapes V_θ — set to False explicitly when that behaviour is desired.
    # Fix a functional coefficient to a value (no gradient). None = trainable
    # (init from entropy_coeff / potential_coeff / interaction_coeff above).
    freeze_entropy: float | None = None
    freeze_potential: float | None = None
    freeze_interaction: float | None = None
    # If True, the potential V is conditioned on time: V_θ(x, t).
    # Implemented for the Stitching model only.
    time_conditioned_potential: bool = False

    # Observation protocol (synthetic experiments)
    obs_protocol: str = (
        "dense"  # observation protocol: "dense", "terminal", "irregular", "trickle"
    )
    obs_n_per_snapshot: int | None = (
        None  # cells per snapshot (None = use all); used by "trickle"
    )
    obs_snapshot_ratio: float = (
        1.0  # fraction of interior snapshots to keep; used by "irregular"
    )

    # McKean-Vlasov on-the-fly data generation (dataset == "mckean-vlasov")
    mv_preset: dict[str, Any] | str | None = field(
        default=None, hash=False, compare=False
    )  # McKean–Vlasov preset (inline dict or JSON path)

    # Synthetic SDE data generation
    synth_n_particles: int = 2000  # number of SDE particles
    synth_n_timesteps: int = 5  # number of observation intervals (T in traj)
    synth_dt: float = 0.01  # observation-interval size (total time = T * dt)
    synth_n_substeps: int = 1  # SDE substeps per observation interval
    synth_beta: float = 0.0  # ground-truth entropy coefficient for simulation diffusion
    synth_train_fraction: float = 0.5  # train fraction for synthetic data
    synth_init: str = "uniform"  # initial distribution: "uniform" or "normal"
    synth_init_loc: float | tuple[float, ...] = (
        0.0  # center of initial distribution: scalar (broadcast) or per-dim vector
    )
    synth_init_scale: float = 4.0  # half-width (uniform) or std dev (normal)
    # Irregular observation times. When set, the SDE is simulated on a fine grid
    # (step = synth_dt) and subsampled to the nearest fine step for each time —
    # overriding the uniform synth_n_timesteps × synth_dt schedule. Coupled only.
    synth_snapshot_times: tuple[float, ...] | None = None
    synth_coupled: bool = True  # if True, all snapshots share the same particle trajectories (jkonet-star "paired" convention; iJKOnet calls this a bug); if False, each snapshot is an independent fresh simulation from t=0 to k·dt — true population-dynamics ("unpaired") setup.

    # Timepoint selection (RNA only)
    # Index into sorted unique timepoints (0=t0, 1=t1, …).  None means use all.
    train_timepoint_idx: tuple[int, ...] | None = None  # timepoints to train on
    eval_timepoint_idx: tuple[int, ...] | None = (
        None  # timepoints to evaluate on (held out)
    )

    # Evaluation
    eval_num_samples: int | None = (
        None  # samples to draw for evaluation (None = use particles directly)
    )
    eval_reps: int = 5  # number of evaluation repetitions for EMD

    # Device
    cuda: bool = False  # use CUDA GPU (vs CPU)

    # Lightspeed (JKOnet*) hyperparameters
    lightspeed_coupling: str = (
        "ot"  # snapshot-pair coupling: "ot" (Hungarian on observed snapshots)
        # or "natural" (use the per-time particle index, only valid
        # when data was simulated with coupled trajectories).
    )

    # Output / display
    output_dir: str = "results"  # root directory for saving results
    plot: bool = True  # generate visualisation (--noplot to skip)
    metric: str = "emd"  # evaluation metric: "emd" or "w2"

    # Set automatically when loaded from a file. Excluded from hash/compare
    # so post-construction stamping by `load_config` doesn't perturb the
    # JIT cache key.
    config_path: str | None = field(default=None, hash=False, compare=False)
    # Explicit experiment name. ``None`` means "fall back to potential/dataset"
    # at read time. The loader stamps the data-preset stem here when more
    # specific than the bare potential/dataset (e.g. ``doublewell-terminal``).
    _experiment_name: str | None = field(default=None, hash=False, compare=False)

    @property
    def mode(self) -> Literal["blackbox", "synthetic"]:
        """Experiment mode, derived from *dataset* / *potential*."""
        if self.potential is not None:
            return "synthetic"
        return "blackbox"

    @property
    def experiment_name(self) -> str:
        """Short human-readable run name, with sensible fallbacks."""
        return self._experiment_name or self.potential or self.dataset or "unknown"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = {k for k in data if not k.startswith("_") and k not in known}
        if unknown:
            raise ValueError(
                f"Unknown config keys: {sorted(unknown)}. "
                f"Did you typo? Known fields: {sorted(known)}"
            )
        filtered: dict[str, Any] = {
            key: value for key, value in data.items() if key in known
        }
        # Auto-tuple-ify list-typed JSON values where the dataclass expects a tuple.
        for tuple_field in (
            "potential_hidden",
            "interaction_hidden",
            "train_timepoint_idx",
            "eval_timepoint_idx",
            "synth_snapshot_times",
        ):
            if filtered.get(tuple_field) is not None:
                filtered[tuple_field] = tuple(filtered[tuple_field])
        # synth_init_loc may be a scalar OR a per-dim vector; tuple-ify lists only.
        if isinstance(filtered.get("synth_init_loc"), list):
            filtered["synth_init_loc"] = tuple(filtered["synth_init_loc"])
        cfg = cls(**filtered)
        if cfg.dataset is not None and cfg.potential is not None:
            raise ValueError(
                "Set either --dataset (blackbox) or --potential (synthetic), not both."
            )
        if cfg.dataset is None and cfg.potential is None:
            raise ValueError(
                "Config must set either --dataset (blackbox) or --potential (synthetic)."
            )
        cfg._warn_misplaced_method_fields(data)
        return cfg

    def _warn_misplaced_method_fields(self, data: dict[str, Any]) -> None:
        """Warn when a method-specific field is set to a non-default for the wrong
        model — the field is silently ignored by the active model, which is the
        class of bug that let presets diverge from what actually ran. Compares
        against defaults so archived ``cfg.json`` (which stores every field at its
        default) does not warn.

        Field ownership comes from the model registry (the single source of
        truth), imported lazily here because the registry lives in the higher
        ``stitching.build`` layer — a module-level import would invert the
        config→build dependency and cycle through ``stitching.config``.
        """
        from stitching.build import model_only_fields

        owners = model_only_fields()
        own = set(owners.get(self.model, ()))
        defaults = Config()

        def nondefault(name: str) -> bool:
            return name in data and getattr(self, name) != getattr(defaults, name)

        # Fields owned by some other model but set here (to a non-default).
        misplaced = {
            f
            for owner, fields in owners.items()
            if owner != self.model
            for f in fields
            if f not in own and nondefault(f)
        }
        if misplaced:
            warnings.warn(
                f"Config sets {sorted(misplaced)} but model={self.model!r}; "
                "these fields are ignored by this model (likely a misplaced "
                "override).",
                stacklevel=3,
            )

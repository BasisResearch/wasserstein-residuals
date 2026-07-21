"""McCann trajectory-init ablation — the headline suite reseeded with McCann.

Appendix ablation: rerun each headline benchmark with
``trajectory_init="mccann"`` (Bures–Wasserstein / OT-geodesic Gaussian seeding)
instead of the default ``"ot"``, so the two seedings can be compared. Each
headline ``ExperimentSpec`` is mirrored into a ``<name>_mccann`` experiment that
reuses the base's configs, variant names, seeds, smoke settings, and eval/plot
hooks — only ``trajectory_init`` differs. The new name gives each ablation its
own ``results/<name>_mccann/`` tree, so the existing OT results are never
touched (checkpoint/figure paths key on experiment name + variant name, not on
``trajectory_init``).

Compare McCann against the *existing* OT results: OT numbers from
``results/<exp>/metrics/*.csv`` vs McCann from ``results/<exp>_mccann/metrics/*.csv``
(identical schema), and each OT figure beside its McCann counterpart.

``wavy_valley`` is excluded — it is already covered by the ``wavy_valley_init``
OT-vs-McCann experiment; ``doublewell`` (scratch) is excluded too.
"""

from __future__ import annotations

from experiments.defs import chiral, cis, rna, synthetic
from experiments.registry import ExperimentSpec, Variant, register

# The headline benchmarks to reseed. wavy_valley is covered by wavy_valley_init;
# doublewell is scratch — both excluded.
_BASES = (synthetic.SPEC, rna.SPEC, cis.SPEC, chiral.SPEC)


def _is_stitching(variant: Variant) -> bool:
    """True if this cell trains the Stitching model (the only one with an init).

    ``trajectory_init`` is a Stitching-owned field; setting it on a lightspeed
    cell would trip the misplaced-override warning. The composite config string
    is ``"<data>:<method>[:<experiment>]"``, so the method is its second field.
    """
    parts = variant.config.split(":")
    return len(parts) < 2 or parts[1] == "stitching"


def _mccann_spec(base: ExperimentSpec) -> ExperimentSpec:
    """Mirror *base* into its McCann-init ablation twin.

    Copies every Stitching variant with ``trajectory_init="mccann"`` layered on
    top of its existing overrides, and reuses the base's seeds, smoke settings,
    and eval/plot hooks verbatim. The distinct ``<name>_mccann`` name isolates
    the output tree from the base experiment.
    """
    variants = tuple(
        Variant(v.name, v.config, {**v.overrides, "trajectory_init": "mccann"})
        for v in base.variants
        if _is_stitching(v)
    )
    return ExperimentSpec(
        name=f"{base.name}_mccann",
        description=f"{base.description} [McCann trajectory-init ablation]",
        variants=variants,
        seeds=base.seeds,
        smoke_epochs=base.smoke_epochs,
        smoke_overrides=base.smoke_overrides,
        smoke_variants=base.smoke_variants,
        evaluate=base.evaluate,
        plots=base.plots,
    )


for _base in _BASES:
    register(_mccann_spec(_base))

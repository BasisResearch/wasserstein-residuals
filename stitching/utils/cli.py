"""Command-line argument parsing for ``python -m stitching.train``."""

from __future__ import annotations

import sys

from stitching.config import (
    Config,
    list_data_presets,
    list_experiment_presets,
    list_method_presets,
    load_config,
)


def parse_args() -> tuple[Config, str | None]:
    """Parse CLI arguments using simple-parsing.

    Returns `(cfg, config_name)`.

    Usage::

        python -m stitching.train -c doublewell:stitching --seed 42
        python -m stitching.train -c rna:stitching
        python -m stitching.train --potential flowers --model stitching
        python -m stitching.train --list-data
        python -m stitching.train --list-methods
        python -m stitching.train --list-experiments
        python -m stitching.train --list-potentials
        python -m stitching.train --list-datasets

    The config preset is given via ``--config`` / ``-c``.
    All ``Config`` fields are available as ``--field-name`` flags
    (underscores also accepted) and override the preset values.
    """
    # Handle informational flags early
    if "--list-potentials" in sys.argv:
        from stitching.synthetic import list_systems

        print("Available systems (use with --potential):")
        for name in list_systems():
            print(f"  {name}")
        sys.exit(0)

    if "--list-datasets" in sys.argv:
        # Imported lazily (it pulls in jax via stitching.data) so the other
        # informational flags stay import-light.
        from stitching.data import DATASET_REGISTRY, list_datasets

        print("Available datasets (use with --dataset):")
        for name in list_datasets():
            print(f"  {name:<16}{DATASET_REGISTRY[name].description}")
        sys.exit(0)

    if "--list-data" in sys.argv:
        print("Available data presets (use with -c <data>:<method>):")
        for name in list_data_presets():
            print(f"  {name}")
        sys.exit(0)

    if "--list-methods" in sys.argv:
        print("Available method presets (use with -c <data>:<method>):")
        for name in list_method_presets():
            print(f"  {name}")
        sys.exit(0)

    if "--list-experiments" in sys.argv:
        print(
            "Available experiment presets "
            "(the 3rd composite tier, use with -c <data>:<method>:<experiment>):"
        )
        for name in list_experiment_presets():
            print(f"  {name}")
        sys.exit(0)

    if "--list-configs" in sys.argv or "--list-combinations" in sys.argv:
        datas = list_data_presets()
        methods = list_method_presets()
        print("Available configurations (use with -c <data>:<method>):\n")
        print(f"  {'DATA':<16}  METHODS")
        for d in datas:
            combos = "  ".join(methods)
            print(f"  {d:<16}  {combos}")
        print(
            f"\n{len(datas)} data × {len(methods)} method "
            f"= {len(datas) * len(methods)} combinations."
        )
        sys.exit(0)

    from simple_parsing import ArgumentParser, DashVariant

    # 1. Extract --config / -c before simple-parsing sees argv.
    config_name: str | None = None
    remaining: list[str] = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        # Handle --config NAME or -c NAME
        if arg in ("--config", "-c"):
            next_idx = i + 2  # +1 for sys.argv offset, +1 for next
            if next_idx < len(sys.argv):
                config_name = sys.argv[next_idx]
                skip_next = True
            continue
        remaining.append(arg)

    # 2. Load preset (if given) so its values become defaults.
    cfg_preset = load_config(config_name)

    # 3. Sanitise config_name for use in output paths (no colons in dirs).
    if config_name is not None and ":" in config_name:
        config_name = config_name.replace(":", "-")

    # 4. Let simple-parsing handle all Config flags.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]] + remaining
    try:
        p = ArgumentParser(
            description="Train a JAX WGF model.",
            add_option_string_dash_variants=DashVariant.UNDERSCORE_AND_DASH,
        )
        p.add_arguments(Config, dest="cfg", default=cfg_preset)
        args = p.parse_args()
    finally:
        sys.argv = saved_argv

    return args.cfg, config_name

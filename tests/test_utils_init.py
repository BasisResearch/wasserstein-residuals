"""Lazy-import contracts for the ``stitching.utils`` package ``__init__``.

Phase-C made the package import-light: the convenience re-exports
(``train`` / ``wasserstein`` / ``parse_args``) resolve lazily via :pep:`562`
``__getattr__`` so importing a light submodule (``plotting`` / ``paper_plots``)
or the package root no longer eagerly drags in the jax/optax-heavy ``optim`` and
``metrics`` modules. These pin that contract — the deliverable of the phase —
since nothing else asserts it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_lazy_reexports_still_resolve() -> None:
    # set_random_seed is eager (defined in __init__); the other three resolve
    # lazily through __getattr__. All must remain importable from the root.
    import stitching.utils as u

    assert callable(u.set_random_seed)
    assert callable(u.train)
    assert callable(u.wasserstein)
    assert callable(u.parse_args)
    # __dir__ advertises the lazy names (tab-completion / introspection).
    assert {"train", "wasserstein", "parse_args"} <= set(dir(u))


def test_getattr_unknown_raises_attribute_error() -> None:
    # The hand-written __getattr__ control flow must fail as a normal missing
    # attribute (so hasattr / from-import / pickling behave), not KeyError.
    import stitching.utils as u

    with pytest.raises(AttributeError, match="has no attribute 'definitely_absent'"):
        u.definitely_absent  # noqa: B018
    assert not hasattr(u, "definitely_absent")


@pytest.mark.parametrize(
    "module", ["stitching.utils.plotting", "stitching.utils.paper_plots"]
)
def test_plot_layer_imports_without_jax(module: str) -> None:
    # The Phase-C deliverable: importing the numpy/matplotlib plotting layer must
    # not pull in jax. Run in a fresh interpreter because jax is already loaded
    # in the main test process (so an in-process check could not detect it).
    code = (
        "import sys\n"
        f"import {module}\n"
        "assert 'jax' not in sys.modules, 'importing %s dragged in jax'\n"
        "print('ok')\n"
    ) % module
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"

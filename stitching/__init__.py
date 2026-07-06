"""Experiment harness for Wasserstein gradient flow models.

Submodules expose the public surface; this package's ``__init__`` only
carries ``__version__``. Use direct submodule imports::

    from stitching.config import Config, load_config
    from stitching.data import load_data
    from stitching.build import build_model, fit, make_loss_fn
    from stitching.evaluate import run_blackbox_evaluation
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stitching")
except PackageNotFoundError:  # running from a checkout without install
    __version__ = "0+unknown"

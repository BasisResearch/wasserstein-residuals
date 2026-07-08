"""Unit tests for the shared figure primitives in ``stitching.utils.paper_plots``.

The 2-D paper runners (cis, chiral, wavy_valley, rna) route their panel
mechanics through this module, so its pixel-affecting contracts are pinned
here: the snapshot binning, the two framing modes (isotropic vs per-axis, with
the degenerate-span floor), and that the matplotlib render helpers touch only the
axes/artists they should. The runners themselves stay verified by the
figure-pixel diff in the Phase-2 verification, not here.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless; no display needed for the render-helper tests
import matplotlib.pyplot as plt  # noqa: E402

from stitching.utils.paper_plots import (  # noqa: E402
    all_snap_times,
    blank_axes,
    filled_contour,
    frame_limits,
    pool_observations,
    scatter_row,
    snapshot_panels,
    trajectory_overlay,
)


class _Data:
    """Minimal SpatioTemporalData stand-in: just the ``.t`` / ``.x`` fields."""

    def __init__(self, t: list[float], x: list[list[float]] | None = None) -> None:
        self.t = np.asarray(t, dtype=np.float32)
        self.x = (
            np.asarray(x, dtype=np.float32)
            if x is not None
            else np.zeros((len(t), 2), dtype=np.float32)
        )


# ---------------------------------------------------------------------------
# pool_observations / all_snap_times — the shared snapshot-panel preamble
# ---------------------------------------------------------------------------


def test_pool_observations_concatenates_train_then_test() -> None:
    train = _Data([0.0, 1.0], [[0.0, 0.0], [1.0, 1.0]])
    test = _Data([2.0], [[2.0, 2.0]])
    obs_t, obs_x = pool_observations(train, test)
    assert obs_t.tolist() == [0.0, 1.0, 2.0]  # train rows first, then test
    assert obs_x.tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


def test_pool_observations_preserves_float32_dtype() -> None:
    # The framing helpers rely on float32 staying float32 (see frame_limits dtype
    # test); pooling must not upcast.
    obs_t, obs_x = pool_observations(_Data([0.0]), _Data([1.0]))
    assert obs_t.dtype == np.float32
    assert obs_x.dtype == np.float32


def test_all_snap_times_sorts_unique_pooled_times() -> None:
    train = _Data([2.0, 0.0, 2.0])
    test = _Data([1.0, 0.0])
    out = all_snap_times(train, test)
    assert out == [0.0, 1.0, 2.0]
    assert all(isinstance(t, float) for t in out)


# ---------------------------------------------------------------------------
# snapshot_panels — observation binning
# ---------------------------------------------------------------------------


def test_snapshot_panels_bins_by_time_with_float_keys() -> None:
    obs_t = np.array([0.0, 0.0, 1.0])
    obs_x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    panels = snapshot_panels(obs_t, obs_x, [0.0, 1.0])
    assert [tv for tv, _ in panels] == [0.0, 1.0]
    assert all(isinstance(tv, float) for tv, _ in panels)  # keys normalised to float
    assert panels[0][1].shape == (2, 2)
    assert panels[1][1].shape == (1, 2)


def test_snapshot_panels_atol_excludes_near_misses() -> None:
    obs_t = np.array([0.0, 0.0009])
    obs_x = np.array([[1.0, 1.0], [2.0, 2.0]])
    # Default atol=1e-6 separates the two; a loose atol pools them.
    assert snapshot_panels(obs_t, obs_x, [0.0])[0][1].shape == (1, 2)
    assert snapshot_panels(obs_t, obs_x, [0.0], atol=1e-2)[0][1].shape == (2, 2)


def test_snapshot_panels_no_match_returns_empty_panel() -> None:
    # A snapshot time matching nothing yields an empty (0, D) panel, not a
    # dropped column — every downstream helper relies on that shape contract.
    panels = snapshot_panels(np.array([0.0]), np.zeros((1, 2)), [9.0])
    assert panels[0][0] == 9.0
    assert panels[0][1].shape == (0, 2)


# ---------------------------------------------------------------------------
# frame_limits — the two padding modes + degenerate-span floor
# ---------------------------------------------------------------------------


def test_frame_limits_isotropic_pads_both_axes_by_larger_span() -> None:
    # span_x=2, span_y=4 → isotropic pad = 0.1 * max(2, 4) = 0.4 on both axes.
    xlim, ylim = frame_limits([np.array([[0.0, 0.0], [2.0, 4.0]])], pad_frac=0.1)
    assert xlim == pytest.approx((-0.4, 2.4))
    assert ylim == pytest.approx((-0.4, 4.4))


def test_frame_limits_per_axis_pads_each_by_own_span() -> None:
    xlim, ylim = frame_limits(
        [np.array([[0.0, 0.0], [2.0, 4.0]])], pad_frac=0.1, per_axis=True
    )
    assert xlim == pytest.approx((-0.2, 2.2))  # 0.1 * span_x=2
    assert ylim == pytest.approx((-0.4, 4.4))  # 0.1 * span_y=4


def test_frame_limits_min_span_floors_degenerate_pad() -> None:
    # A single point has zero span; min_span keeps the window from collapsing.
    xlim, ylim = frame_limits([np.array([[1.0, 1.0]])], pad_frac=0.1, min_span=2.0)
    assert xlim == pytest.approx((0.8, 1.2))
    assert ylim == pytest.approx((0.8, 1.2))


def test_frame_limits_ignores_columns_beyond_first_two() -> None:
    # A huge third column must not widen the frame (only cols 0,1 are used).
    pts = np.array([[0.0, 0.0, 1e9], [1.0, 1.0, -1e9]])
    xlim, ylim = frame_limits([pts], pad_frac=0.0)
    assert xlim == pytest.approx((0.0, 1.0))
    assert ylim == pytest.approx((0.0, 1.0))


def test_frame_limits_concatenates_multiple_arrays() -> None:
    a = np.array([[0.0, 0.0]])
    b = np.array([[10.0, -5.0]])
    xlim, ylim = frame_limits([a, b], pad_frac=0.0)
    assert xlim == pytest.approx((0.0, 10.0))
    assert ylim == pytest.approx((-5.0, 0.0))


def test_frame_limits_preserves_input_dtype_in_endpoints() -> None:
    # The load-bearing invariant: endpoints are NOT float()-cast, so float32 data
    # frames in float32. A float() upcast would shift the limits ~1e-7 (NEP50 weak
    # promotion) and move rasterized lines sub-pixel — a silently different figure.
    pts32 = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    xlim, ylim = frame_limits([pts32], pad_frac=0.1)
    assert xlim[0].dtype == np.float32
    assert ylim[1].dtype == np.float32
    # The per_axis path is the one the trajectory figures use (and where the
    # float32 framing bug originally surfaced), so pin its dtype too.
    xlim_pa, ylim_pa = frame_limits([pts32], pad_frac=0.08, per_axis=True)
    assert xlim_pa[0].dtype == np.float32
    assert ylim_pa[1].dtype == np.float32
    # float64 input stays float64 (no spurious downcast either).
    xlim64, _ = frame_limits([pts32.astype(np.float64)], pad_frac=0.1)
    assert xlim64[0].dtype == np.float64


def test_frame_limits_empty_iterable_raises() -> None:
    with pytest.raises(ValueError, match="at least one point array"):
        frame_limits([])


def test_frame_limits_all_empty_panels_raises() -> None:
    # The reachable degenerate case: every snapshot matched zero observations.
    with pytest.raises(ValueError, match="non-empty"):
        frame_limits([np.empty((0, 2))])


def test_frame_limits_tolerates_a_mix_of_empty_and_nonempty() -> None:
    # The realistic case — one snapshot populated, one empty — must still frame.
    xlim, ylim = frame_limits([np.empty((0, 2)), np.array([[0.0, 0.0]])], pad_frac=0.0)
    assert xlim == pytest.approx((0.0, 0.0))
    assert ylim == pytest.approx((0.0, 0.0))


# ---------------------------------------------------------------------------
# scatter_row — surplus axes untouched; first two columns only
# ---------------------------------------------------------------------------


def test_scatter_row_leaves_surplus_axes_untouched() -> None:
    fig, axes = plt.subplots(1, 3)
    panels = [(0.0, np.zeros((4, 2))), (1.0, np.ones((4, 2)))]
    scatter_row(axes, panels, color="tab:gray")
    assert len(axes[0].collections) == 1
    assert len(axes[1].collections) == 1
    assert len(axes[2].collections) == 0  # trailing comparison column untouched
    plt.close(fig)


def test_scatter_row_plots_first_two_columns() -> None:
    fig, axes = plt.subplots(1, 1, squeeze=False)
    pts = np.array([[1.0, 2.0, 99.0], [3.0, 4.0, 99.0]])
    scatter_row(axes[0], [(0.0, pts)], color="k")
    offsets = axes[0, 0].collections[0].get_offsets()
    assert np.allclose(offsets, [[1.0, 2.0], [3.0, 4.0]])
    plt.close(fig)


def test_scatter_row_too_few_axes_raises() -> None:
    # Fewer axes than panels would silently drop trailing panels — guard loudly.
    fig, axes = plt.subplots(1, 1, squeeze=False)
    panels = [(0.0, np.zeros((2, 2))), (1.0, np.zeros((2, 2)))]
    with pytest.raises(ValueError, match="dropped silently"):
        scatter_row(axes[0], panels, color="k")
    plt.close(fig)


# ---------------------------------------------------------------------------
# blank_axes — limits / ticks / aspect / grid
# ---------------------------------------------------------------------------


def test_blank_axes_sets_limits_and_clears_ticks() -> None:
    fig, axes = plt.subplots(1, 2)
    blank_axes(axes, (-1.0, 1.0), (-2.0, 2.0))
    for ax in axes:
        assert ax.get_xlim() == (-1.0, 1.0)
        assert ax.get_ylim() == (-2.0, 2.0)
        assert list(ax.get_xticks()) == []
        assert list(ax.get_yticks()) == []
    plt.close(fig)


def test_blank_axes_aspect_and_grid_are_opt_in() -> None:
    fig, (ax_plain, ax_styled) = plt.subplots(1, 2)
    blank_axes([ax_plain], (0.0, 1.0), (0.0, 1.0))
    blank_axes([ax_styled], (0.0, 1.0), (0.0, 1.0), grid=True, aspect="equal")
    assert ax_plain.get_aspect() == "auto"
    assert ax_plain.get_axisbelow() is not True  # grid not requested
    assert ax_styled.get_aspect() in (1.0, "equal")  # "equal" (value varies by mpl)
    assert ax_styled.get_axisbelow() is True  # grid requested
    plt.close(fig)


# ---------------------------------------------------------------------------
# trajectory_overlay — particle-path cap + first two coords
# ---------------------------------------------------------------------------


def test_trajectory_overlay_caps_number_of_paths() -> None:
    fig, ax = plt.subplots()
    traj = np.zeros((5, 10, 2))  # (T=5, N=10, D=2)
    trajectory_overlay(ax, traj, color="g", n_show=3)
    assert len(ax.lines) == 3  # capped to n_show
    plt.close(fig)


def test_trajectory_overlay_shows_all_when_fewer_than_cap() -> None:
    fig, ax = plt.subplots()
    traj = np.zeros((4, 2, 3))  # N=2 < n_show; D=3 (only first 2 used)
    trajectory_overlay(ax, traj, color="g", n_show=100)
    assert len(ax.lines) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# filled_contour — renders both a filled and a line contour
# ---------------------------------------------------------------------------


def test_filled_contour_draws_fill_and_lines() -> None:
    fig, ax = plt.subplots()
    xs = np.linspace(-1.0, 1.0, 8)
    X, Y = np.meshgrid(xs, xs)
    Z = X**2 + Y**2
    filled_contour(ax, X, Y, Z, vmin=0.0, vmax=2.0)
    # Both the contourf fill AND the contour line set register collections;
    # deleting either call drops the count, so >=2 pins "fill and lines".
    assert len(ax.collections) >= 2
    plt.close(fig)


def test_filled_contour_constant_field_does_not_raise() -> None:
    # A flat learned field (vmin == vmax) must not blow up the contour call.
    fig, ax = plt.subplots()
    xs = np.linspace(0.0, 1.0, 5)
    X, Y = np.meshgrid(xs, xs)
    Z = np.ones_like(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # mpl warns "no contour levels"
        filled_contour(ax, X, Y, Z, vmin=1.0, vmax=1.0)
    plt.close(fig)

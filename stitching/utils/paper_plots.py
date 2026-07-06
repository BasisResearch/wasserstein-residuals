"""Pure numpy/matplotlib panel primitives shared by the experiment ``defs/`` plots.

Each experiment's plot hook used to carry a monolithic ``plot()`` that recomputed
the same snapshot-panel scaffolding — pool train+test observations, bin them by
snapshot time, frame the axes to a common window, scatter each panel, blank the
ticks — plus trajectory and learned-potential contour overlays. Those mechanics
live here as small, side-effect-free helpers that take already-evaluated numpy
arrays and a matplotlib ``Axes`` and render: no jax, no model, no IO. The plot
hooks keep only their experiment-specific extraction (``model.sample``,
``jax.vmap`` over the potential) and figure assembly.

This sits in the installed ``stitching`` package (like ``utils/runners.py``) so
the helpers are unit-testable from ``tests/`` and covered by ``--doctest-modules``.
It imports numpy only — the matplotlib ``Axes`` type is import-guarded behind
``TYPE_CHECKING`` — so, unlike ``utils/plotting.py``, importing it never drags
jax (or even matplotlib) into a caller's import graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes

    from stitching._kde import SpatioTemporalData

# One snapshot panel: its time and the ``(N, D)`` points observed at that time.
Panel = tuple[float, np.ndarray]


def pool_observations(
    train_data: SpatioTemporalData, test_data: SpatioTemporalData
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate train + test observations into pooled ``(times, coords)`` arrays.

    The first step every snapshot-panel plot shares: pool the train and held-out
    snapshots so they bin together. ``np.asarray`` realises the jax-backed fields
    as numpy; no jax import is needed (the array protocol handles it), so this
    stays in the jax-free plotting layer.

    Args:
        train_data: Training snapshots.
        test_data: Held-out snapshots.

    Returns:
        ``(obs_t, obs_x)`` — pooled times ``(M,)`` and coordinates ``(M, D)``.
    """
    obs_t = np.concatenate([np.asarray(train_data.t), np.asarray(test_data.t)])
    obs_x = np.concatenate([np.asarray(train_data.x), np.asarray(test_data.x)])
    return obs_t, obs_x


def all_snap_times(
    train_data: SpatioTemporalData, test_data: SpatioTemporalData
) -> list[float]:
    """Sorted unique observation times across the pooled train + test snapshots.

    Args:
        train_data: Training snapshots.
        test_data: Held-out snapshots.

    Returns:
        Ascending list of distinct observation times (as Python floats).

    >>> import numpy as np
    >>> class D:  # minimal stand-in for SpatioTemporalData
    ...     def __init__(self, t):
    ...         self.t = np.asarray(t)
    ...         self.x = np.zeros((len(t), 2))
    >>> all_snap_times(D([2.0, 0.0, 2.0]), D([1.0, 0.0]))
    [0.0, 1.0, 2.0]
    """
    obs_t, _ = pool_observations(train_data, test_data)
    return sorted(float(t) for t in np.unique(obs_t))


def snapshot_panels(
    obs_t: np.ndarray,
    obs_x: np.ndarray,
    snap_t: Sequence[float],
    *,
    atol: float = 1e-6,
) -> list[Panel]:
    """Bin pooled observations into one ``(time, points)`` panel per snapshot.

    Args:
        obs_t: Pooled observation times, shape ``(M,)``.
        obs_x: Pooled observation coordinates, shape ``(M, D)``.
        snap_t: Snapshot times to extract a panel for.
        atol: Absolute tolerance for matching an observation to a snapshot time.

    Returns:
        ``[(t, pts), ...]`` where ``pts`` are the ``obs_x`` rows whose time is
        within *atol* of ``t``.

    >>> import numpy as np
    >>> t = np.array([0.0, 0.0, 1.0])
    >>> x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> [(tv, p.shape) for tv, p in snapshot_panels(t, x, [0.0, 1.0])]
    [(0.0, (2, 2)), (1.0, (1, 2))]
    """
    obs_t = np.asarray(obs_t)
    obs_x = np.asarray(obs_x)
    return [(float(t), obs_x[np.isclose(obs_t, t, atol=atol)]) for t in snap_t]


def frame_limits(
    point_arrays: Iterable[np.ndarray],
    *,
    pad_frac: float = 0.1,
    min_span: float = 0.0,
    per_axis: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Padded ``(xlim, ylim)`` enclosing the first two columns of every array.

    Args:
        point_arrays: Iterable of ``(N, 2+)`` arrays; only columns 0 and 1 used.
        pad_frac: Margin as a fraction of the data span.
        min_span: Floor on the span used to size the margin (guards a degenerate
            single-point / near-uniform panel against a zero-width pad).
        per_axis: If True, pad each axis by ``pad_frac`` of its own span; if
            False, pad both axes by ``pad_frac`` of the larger span (isotropic).

    Returns:
        ``((xmin, xmax), (ymin, ymax))``. Each endpoint keeps the input arrays'
        dtype (a numpy scalar): the limits are passed straight to ``ax.set_xlim``,
        and *not* upcasting float32 data to float64 keeps the rendered figures
        bit-identical to the hand-rolled framing the runners used before.

    Raises:
        ValueError: if *point_arrays* is empty or holds no points at all (a clear
            domain error instead of a cryptic empty-reduction failure downstream).

    >>> xlim, ylim = frame_limits([np.array([[0.0, 0.0], [2.0, 4.0]])], pad_frac=0.0)
    >>> (float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
    (0.0, 2.0, 0.0, 4.0)
    """
    arrays = [np.asarray(a) for a in point_arrays]
    if not arrays:
        raise ValueError("frame_limits requires at least one point array.")
    xs = np.concatenate([a[:, 0].ravel() for a in arrays])
    ys = np.concatenate([a[:, 1].ravel() for a in arrays])
    if xs.size == 0:
        raise ValueError("frame_limits requires at least one non-empty point array.")
    ptp_x, ptp_y = float(np.ptp(xs)), float(np.ptp(ys))
    if per_axis:
        padx = pad_frac * max(ptp_x, min_span)
        pady = pad_frac * max(ptp_y, min_span)
    else:
        padx = pady = pad_frac * max(ptp_x, ptp_y, min_span)
    # NB: endpoints are intentionally left as numpy scalars (not float()-cast) so
    # float32 inputs frame exactly as the original inline ``arr.min() - pad`` did.
    return (
        (xs.min() - padx, xs.max() + padx),
        (ys.min() - pady, ys.max() + pady),
    )


def scatter_row(
    axes_row: Sequence[Axes],
    panels: Sequence[Panel],
    *,
    color: str,
    s: float = 2.0,
    alpha: float = 0.6,
    rasterized: bool = True,
) -> None:
    """Scatter each panel's points onto the matching axis in *axes_row*.

    Panel ``j`` (its ``(N, 2+)`` point array) is drawn on ``axes_row[j]`` using
    columns 0 and 1; surplus axes (e.g. a trailing comparison column) are left
    untouched. Titles and limits are the caller's responsibility.

    Args:
        axes_row: Axes for this row; may be longer than *panels* (a trailing
            comparison column is left untouched) but never shorter.
        panels: ``(time, points)`` panels, one per column.
        color: Scatter colour.
        s: Marker size.
        alpha: Marker opacity.
        rasterized: Rasterize the scatter (keeps vector output light).

    Raises:
        ValueError: if there are fewer axes than panels (which would silently
            drop the trailing panels — almost always a miscounted-column bug).
    """
    if len(panels) > len(axes_row):
        raise ValueError(
            f"scatter_row got {len(panels)} panels but only {len(axes_row)} "
            "axes; the extra panels would be dropped silently."
        )
    for ax, (_t, pts) in zip(axes_row, panels, strict=False):
        ax.scatter(
            pts[:, 0], pts[:, 1], s=s, c=color, alpha=alpha, rasterized=rasterized
        )


def blank_axes(
    axes: Iterable[Axes],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    grid: bool = False,
    aspect: str | None = None,
) -> None:
    """Apply shared snapshot-axis styling: fixed limits, no ticks, optional grid.

    Args:
        axes: Axes to style (any iterable, e.g. ``axes[:, :n].ravel()``).
        xlim: Shared x-limits.
        ylim: Shared y-limits.
        grid: Draw the light grey reference grid (``set_axisbelow`` + grid).
        aspect: If given, passed to ``ax.set_aspect`` (e.g. ``"equal"``).
    """
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        if aspect is not None:
            ax.set_aspect(aspect)
        if grid:
            ax.set_axisbelow(True)
            ax.grid(True, color="0.6", alpha=0.35, lw=0.4)


def trajectory_overlay(
    ax: Axes,
    traj: np.ndarray,
    *,
    color: str,
    n_show: int = 100,
    lw: float = 0.6,
    alpha: float = 0.5,
    zorder: float = 5,
) -> None:
    """Overlay up to *n_show* particle paths from a ``(T, N, D)`` trajectory.

    Draws ``min(n_show, N)`` evenly-indexed particle columns as 2-D lines (first
    two coordinates) on *ax*.

    Args:
        ax: Target axis.
        traj: Trajectory array, shape ``(T, N, D)`` with ``D >= 2``.
        color: Line colour.
        n_show: Cap on the number of particle paths drawn.
        lw: Line width.
        alpha: Line opacity.
        zorder: Draw order.
    """
    traj = np.asarray(traj)
    n = traj.shape[1]
    idx = np.linspace(0, n - 1, min(n_show, n)).astype(int)
    for k in idx:
        ax.plot(
            traj[:, k, 0], traj[:, k, 1], color=color, lw=lw, alpha=alpha, zorder=zorder
        )


def filled_contour(
    ax: Axes,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    levels: int = 22,
    line_levels: int = 10,
    cmap: str = "RdYlBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.55,
    line_color: str = "k",
    line_alpha: float = 0.18,
    line_width: float = 0.4,
) -> None:
    """Filled contour of *Z* with a faint overlaid line contour of the same field.

    The shared idiom for the learned-potential panels: a colour ``contourf`` plus
    thin ``contour`` level lines.

    Args:
        ax: Target axis.
        X: Meshgrid x-coordinates.
        Y: Meshgrid y-coordinates.
        Z: Field values on the grid.
        levels: Filled-contour level count.
        line_levels: Line-contour level count.
        cmap: Colormap for the filled contour.
        vmin: Lower colour limit (``None`` lets matplotlib autoscale).
        vmax: Upper colour limit.
        alpha: Fill opacity.
        line_color: Level-line colour.
        line_alpha: Level-line opacity.
        line_width: Level-line width.
    """
    ax.contourf(X, Y, Z, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    ax.contour(
        X,
        Y,
        Z,
        levels=line_levels,
        colors=line_color,
        alpha=line_alpha,
        linewidths=line_width,
    )

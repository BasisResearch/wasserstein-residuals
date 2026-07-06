"""Plotting utilities for JAX WGF experiments.

Produces:
- Per-timepoint scatter plots (first two PCA components) comparing
  model samples to observed data, with potential contour overlay.
- An MP4 animation of particle evolution over continuous time.

Import-light by design: only :func:`visualise` evaluates the model (the sole
jax-touching path), so it imports jax lazily in its body, and the jax /
``SpatioTemporalData`` *annotations* are guarded behind ``TYPE_CHECKING`` (the
``_kde`` package itself pulls in jax). Importing this module therefore drags in
matplotlib + numpy but **not** jax — so the ``make_trajectory_callback``
training-callback path, and any future numpy-only caller, doesn't pay the jax
import. Mirrors the jax-free guarantee :mod:`stitching.utils.paper_plots`
already makes.
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.animation as animation
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import jax

    from stitching._kde import SpatioTemporalData

# ------------------------------------------------------------------
# Scatter + potential contour plots
# ------------------------------------------------------------------


def plot_snapshots(
    samples_by_time: dict[float, jax.Array],
    test_data: SpatioTemporalData,
    potential_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    save_path: str | Path | None = None,
    title: str = "Model vs Observed",
    figsize_per_panel: tuple[float, float] = (4.0, 3.5),
    contour_grid: int = 60,
) -> Figure:
    """Plot model density vs observed data at each timepoint.

    Model density at each time is rendered as a histogram of the samples
    in ``samples_by_time[t]`` (KDE draws for Stitching, JKO-rollout samples
    for Lightspeed). Use plenty of samples (≥10k) to get smooth histograms.

    Parameters
    ----------
    samples_by_time:
        `{time: (N, D)}` dict returned by `model.sample()`.
    test_data:
        Ground-truth spatiotemporal data (all timepoints).
    potential_fn:
        Callable `(M, 2) -> (M,)` evaluating the learned potential on
        2-D coordinates.  If provided, coloured contours are drawn.
    save_path:
        If given, saves the figure to this path (PNG).
    title:
        Super-title for the figure.
    figsize_per_panel:
        Width × height of each subplot.
    contour_grid:
        Number of grid points per axis for the potential contour.

    Returns
    -------
    The matplotlib `Figure`.
    """
    all_times = sorted(samples_by_time.keys())
    # Cap at 2x8=16 panels; for dense grids (e.g. 1000-snapshot trickle),
    # pick evenly-spaced indices covering the full time range.
    max_panels = 16
    if len(all_times) > max_panels:
        idx = np.linspace(0, len(all_times) - 1, max_panels).round().astype(int)
        times = [all_times[i] for i in idx]
    else:
        times = all_times
    n_panels = len(times)
    max_cols = 8
    n_cols = min(n_panels, max_cols)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    # Hide unused axes in the last row
    for idx in range(n_panels, len(axes_flat)):  # type: ignore[assignment]
        axes_flat[idx].set_visible(False)

    raw = np.asarray(test_data.x)
    dim = 1 if (raw.ndim == 1 or raw.shape[1] == 1) else raw.shape[1]
    test_t_np = np.asarray(test_data.t).ravel()

    # ------------------------------------------------------------------
    # 1-D branch: KDE density curves + optional potential line
    # ------------------------------------------------------------------
    if dim == 1:
        test_x1d = raw.ravel()
        all_vals = [test_x1d]
        all_vals.extend(np.asarray(pts).ravel() for pts in samples_by_time.values())
        all_cat = np.concatenate(all_vals)
        x_min = float(all_cat.min())
        x_max = float(all_cat.max())
        pad = 0.1 * max(x_max - x_min, 1e-3)
        x_min -= pad
        x_max += pad
        bins = np.linspace(x_min, x_max, 50)

        for i, t_val in enumerate(times):
            ax = axes_flat[i]

            # Observed histogram
            mask = np.isclose(test_t_np, t_val, atol=1e-6)
            obs = test_x1d[mask]
            if obs.shape[0] >= 2:
                ax.hist(
                    obs,
                    bins=bins,
                    density=True,
                    alpha=0.4,
                    color="steelblue",
                    label="observed",
                )

            # Model histogram (KDE / ULA / push-forward samples)
            gen = np.asarray(samples_by_time[t_val]).ravel()
            if gen.shape[0] >= 2:
                ax.hist(
                    gen,
                    bins=bins,
                    density=True,
                    alpha=0.4,
                    color="tomato",
                    label="model",
                )

            ax.set_title(f"t = {t_val:.2f}")
            ax.set_xlabel("x")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_ylabel("density")
                ax.legend(fontsize=7)

        fig.suptitle(title, fontsize=13)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # 2-D branch: scatter plot (original behaviour)
    # ------------------------------------------------------------------
    def _to2d(pts: np.ndarray | jax.Array) -> np.ndarray:
        pts = np.asarray(pts)
        return pts[:, :2]

    test_x_np = raw[:, :2]

    # Gather global bounds for consistent axes
    all_pts = [test_x_np]
    all_pts.extend(_to2d(pts) for pts in samples_by_time.values())
    all_pts_cat = np.concatenate(all_pts, axis=0)
    x_min, x_max = float(all_pts_cat[:, 0].min()), float(all_pts_cat[:, 0].max())
    y_min, y_max = float(all_pts_cat[:, 1].min()), float(all_pts_cat[:, 1].max())
    pad = 0.1 * max(x_max - x_min, y_max - y_min, 1e-3)
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    for i, t_val in enumerate(times):
        ax = axes_flat[i]

        # Potential contour background
        if potential_fn is not None:
            gx = np.linspace(x_min, x_max, contour_grid)
            gy = np.linspace(y_min, y_max, contour_grid)
            GX, GY = np.meshgrid(gx, gy)
            grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
            V = potential_fn(grid_pts).reshape(contour_grid, contour_grid)
            ax.contourf(GX, GY, V, levels=20, cmap="YlOrRd", alpha=0.35)

        # Observed data at this timepoint
        mask = np.isclose(test_t_np, t_val, atol=1e-6)
        obs = test_x_np[mask]
        ax.scatter(
            obs[:, 0], obs[:, 1], s=8, alpha=0.2, c="steelblue", label="observed"
        )

        # Model samples
        gen = _to2d(samples_by_time[t_val])
        ax.scatter(gen[:, 0], gen[:, 1], s=8, alpha=0.2, c="tomato", label="model")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"t = {t_val:.2f}")
        ax.set_xlabel("PC 1")
        if i == 0:
            ax.set_ylabel("PC 2")
            ax.legend(fontsize=7, markerscale=2)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


# ------------------------------------------------------------------
# Animation of particle evolution
# ------------------------------------------------------------------


def animate_particles(
    trajectories: np.ndarray,
    t_grid: np.ndarray,
    test_data: SpatioTemporalData | None = None,
    potential_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    save_path: str | Path | None = None,
    fps: int = 10,
    figsize: tuple[float, float] = (5, 4.5),
    contour_grid: int = 60,
    title: str = "Particle Evolution",
) -> animation.FuncAnimation:
    """Create an animation of particles evolving over time.

    Parameters
    ----------
    trajectories:
        Array of shape `(T, N, 2+)` — first two columns are plotted.
    t_grid:
        Time values `(T,)` corresponding to trajectory frames.
    test_data:
        If provided, observed points are shown as a static background.
    potential_fn:
        If provided, potential contours are drawn.
    save_path:
        If given, saves the animation (MP4 or GIF depending on extension).
    fps:
        Frames per second.
    figsize:
        Figure size.
    contour_grid:
        Grid resolution for contour.
    title:
        Title for the animation.

    Returns
    -------
    `FuncAnimation` object.
    """
    # Cap frame count (subsample evenly across time).
    max_frames = 100
    if trajectories.shape[0] > max_frames:
        idx = np.linspace(0, trajectories.shape[0] - 1, max_frames).round().astype(int)
        trajectories = trajectories[idx]
        t_grid = np.asarray(t_grid)[idx]

    T, _N = trajectories.shape[0], trajectories.shape[1]
    is_1d = trajectories.shape[2] == 1

    # ------------------------------------------------------------------
    # 1-D branch: animated histogram density
    # ------------------------------------------------------------------
    if is_1d:
        traj_1d = trajectories[:, :, 0]  # (T, N)

        # Global x range across all frames + observed data
        x_min = float(traj_1d.min())
        x_max = float(traj_1d.max())
        # Pre-compute per-frame observed histograms (snap to nearest obs time)
        obs_by_frame: list[np.ndarray | None] = [None] * T
        if test_data is not None:
            obs_all_1d = np.asarray(test_data.x).ravel()
            obs_t_np = np.asarray(test_data.t).ravel()
            obs_unique_t = np.unique(obs_t_np)
            x_min = min(x_min, float(obs_all_1d.min()))
            x_max = max(x_max, float(obs_all_1d.max()))
            for f in range(T):
                nearest_idx = int(np.argmin(np.abs(obs_unique_t - t_grid[f])))
                nearest_t = obs_unique_t[nearest_idx]
                mask = np.isclose(obs_t_np, nearest_t, atol=1e-6)
                obs_by_frame[f] = obs_all_1d[mask]
        pad = 0.1 * max(x_max - x_min, 1e-3)
        x_min -= pad
        x_max += pad
        bins = np.linspace(x_min, x_max, 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = float(bins[1] - bins[0])

        # Pre-compute max density for a fixed y-axis
        max_density = max(
            float(np.histogram(traj_1d[f], bins=bins, density=True)[0].max())
            for f in range(T)
        )
        for obs_f in obs_by_frame:
            if obs_f is not None and obs_f.shape[0] >= 2:
                d = float(np.histogram(obs_f, bins=bins, density=True)[0].max())
                max_density = max(max_density, d)

        fig, ax = plt.subplots(figsize=figsize)

        # Observed histogram (animated, initialised to frame 0)
        obs0 = obs_by_frame[0]
        obs_counts0 = (
            np.histogram(obs0, bins=bins, density=True)[0]
            if obs0 is not None and obs0.shape[0] >= 2
            else np.zeros(len(bin_centers))
        )
        obs_bar = ax.bar(
            bin_centers,
            obs_counts0,
            width=bin_width,
            alpha=0.4,
            color="steelblue",
            label="observed",
        )

        # Model particle histogram (animated, initialised to frame 0)
        counts0, _ = np.histogram(traj_1d[0], bins=bins, density=True)
        bar_container = ax.bar(
            bin_centers,
            counts0,
            width=bin_width,
            alpha=0.6,
            color="tomato",
            label="particles",
        )
        time_text = ax.text(
            0.02, 0.95, "", transform=ax.transAxes, fontsize=10, verticalalignment="top"
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, max_density * 1.15)
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
        ax.set_title(title)

        def _init():
            for rect in bar_container.patches:
                rect.set_height(0.0)
            for rect in obs_bar.patches:
                rect.set_height(0.0)
            time_text.set_text("")
            return (*obs_bar.patches, *bar_container.patches, time_text)

        def _update(frame):
            # Update observed histogram
            obs_f = obs_by_frame[frame]
            if obs_f is not None and obs_f.shape[0] >= 2:
                obs_c, _ = np.histogram(obs_f, bins=bins, density=True)
            else:
                obs_c = np.zeros(len(bin_centers))
            for rect, h in zip(obs_bar.patches, obs_c):
                rect.set_height(h)
            # Update model histogram
            counts, _ = np.histogram(traj_1d[frame], bins=bins, density=True)
            for rect, h in zip(bar_container.patches, counts):
                rect.set_height(h)
            time_text.set_text(f"t = {t_grid[frame]:.3f}")
            return (*obs_bar.patches, *bar_container.patches, time_text)

        anim = animation.FuncAnimation(
            fig,
            _update,
            init_func=_init,
            frames=T,
            interval=1000 // fps,
            blit=True,
        )

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ext = save_path.suffix.lower()
            if ext == ".gif":
                writer = animation.PillowWriter(fps=fps)
            else:
                writer = animation.FFMpegWriter(fps=fps, codec="h264")  # type: ignore[assignment]
            anim.save(str(save_path), writer=writer)

        plt.close(fig)
        return anim

    # ------------------------------------------------------------------
    # 2-D branch: animated scatter (original behaviour)
    # ------------------------------------------------------------------
    traj_2d = trajectories[:, :, :2]

    # Global bounds
    x_min, x_max = float(traj_2d[:, :, 0].min()), float(traj_2d[:, :, 0].max())
    y_min, y_max = float(traj_2d[:, :, 1].min()), float(traj_2d[:, :, 1].max())
    if test_data is not None:
        obs_np = np.asarray(test_data.x)
        obs_2d = obs_np[:, :2]
        x_min = min(x_min, float(obs_2d[:, 0].min()))
        x_max = max(x_max, float(obs_2d[:, 0].max()))
        y_min = min(y_min, float(obs_2d[:, 1].min()))
        y_max = max(y_max, float(obs_2d[:, 1].max()))
    pad = 0.1 * max(x_max - x_min, y_max - y_min, 1e-3)
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    fig, ax = plt.subplots(figsize=figsize)

    # Static potential contours
    if potential_fn is not None:
        gx = np.linspace(x_min, x_max, contour_grid)
        gy = np.linspace(y_min, y_max, contour_grid)
        GX, GY = np.meshgrid(gx, gy)
        grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
        V = potential_fn(grid_pts).reshape(contour_grid, contour_grid)
        ax.contourf(GX, GY, V, levels=20, cmap="YlOrRd", alpha=0.3)

    # Pre-compute per-frame observed points (snap to nearest obs time)
    obs_by_frame_2d: list[np.ndarray | None] = [None] * T
    if test_data is not None:
        obs_all_2d = np.asarray(test_data.x)[:, :2]
        obs_t_np = np.asarray(test_data.t).ravel()
        obs_unique_t = np.unique(obs_t_np)
        for f in range(T):
            nearest_idx = int(np.argmin(np.abs(obs_unique_t - t_grid[f])))
            nearest_t = obs_unique_t[nearest_idx]
            mask = np.isclose(obs_t_np, nearest_t, atol=1e-6)
            obs_by_frame_2d[f] = obs_all_2d[mask]

    # Animated observed scatter
    obs_scat = ax.scatter([], [], s=8, alpha=0.3, c="steelblue", label="observed")
    # Animated model scatter
    scat = ax.scatter([], [], s=12, alpha=0.6, c="tomato", label="particles")
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=10, verticalalignment="top"
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=7, markerscale=2, loc="upper right")
    ax.set_title(title)

    def _init_2d():
        obs_scat.set_offsets(np.empty((0, 2)))
        scat.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return obs_scat, scat, time_text

    def _update_2d(frame):
        obs_f = obs_by_frame_2d[frame]
        if obs_f is not None:
            obs_scat.set_offsets(obs_f)
        else:
            obs_scat.set_offsets(np.empty((0, 2)))
        scat.set_offsets(traj_2d[frame])
        time_text.set_text(f"t = {t_grid[frame]:.3f}")
        return obs_scat, scat, time_text

    anim = animation.FuncAnimation(
        fig,
        _update_2d,
        init_func=_init_2d,
        frames=T,
        interval=1000 // fps,
        blit=True,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ext = save_path.suffix.lower()
        if ext == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, codec="h264")  # type: ignore[assignment]
        anim.save(str(save_path), writer=writer)

    plt.close(fig)
    return anim


# ------------------------------------------------------------------
# Trajectory plot (time vs value, 1-D data)
# ------------------------------------------------------------------


def plot_trajectory(
    trajectories: np.ndarray,
    t_grid: np.ndarray,
    test_data: SpatioTemporalData | None = None,
    save_path: str | Path | None = None,
    title: str = "Particle Trajectories",
    max_particles: int = 50,
    alpha_line: float = 0.4,
    alpha_data: float = 0.15,
) -> Figure:
    """Plot particle trajectories as lines over time (1-D data).

    Parameters
    ----------
    trajectories:
        Particle positions `(T, N, D)` with `D=1`.
    t_grid:
        Time values `(T,)` corresponding to trajectory frames.
    test_data:
        Optional observed data shown as a background scatter.
    save_path:
        If given, saves the figure to this path.
    title:
        Figure title.
    max_particles:
        Maximum number of particle lines to draw (subsampled if needed).
    alpha_line:
        Opacity of particle lines.
    alpha_data:
        Opacity of observed data scatter points.
    """
    T, N, D = trajectories.shape
    vals = trajectories[:, :, 0]  # (T, N) — first (only) dimension

    fig, ax = plt.subplots(figsize=(10, 5))

    # Background: observed data scatter
    if test_data is not None:
        obs_t = np.asarray(test_data.t)
        obs_x = np.asarray(test_data.x)
        if obs_x.ndim == 2:
            obs_x = obs_x[:, 0]
        ax.scatter(
            obs_t, obs_x, s=3, c="grey", alpha=alpha_data, label="observed", zorder=1
        )

    # Subsample particles if too many
    if N > max_particles:
        idx = np.linspace(0, N - 1, max_particles, dtype=int)
    else:
        idx = np.arange(N)

    for i in idx:
        ax.plot(
            t_grid, vals[:, i], linewidth=0.7, alpha=alpha_line, color="C0", zorder=2
        )
        ax.scatter(t_grid, vals[:, i], s=8, alpha=alpha_line, color="C0", zorder=3)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    if test_data is not None:
        ax.legend(loc="upper right", markerscale=3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)

    plt.close(fig)
    return fig


def make_trajectory_callback(
    train_data: SpatioTemporalData,
    test_data: SpatioTemporalData,
    callback_every: int = 10,
):
    """Return a (callback, finalize) pair for recording trajectory evolution.

    *callback(model, epoch)* captures a trajectory snapshot.
    *finalize(output_dir, model_name)* stitches frames into
    ``trajectory.mp4`` (its last frame matches ``trajectory.png``).

    Returns ``(None, None)`` when the data isn't 1-D.
    """
    dim = train_data.x.shape[1]
    if dim != 1:
        return None, None

    frames: list[tuple[int, np.ndarray, np.ndarray]] = []

    def _callback(model: object, epoch: int) -> None:
        if not hasattr(model, "trajectories") or not hasattr(model, "t_grid"):
            return
        traj = np.asarray(model.trajectories)  # (T, N, D)
        t_grid = np.asarray(model.t_grid)  # (T,)
        frames.append((epoch, traj.copy(), t_grid.copy()))

    def _finalize(output_dir: str, model_name: str) -> None:
        if not frames:
            return

        out = Path(output_dir) / model_name
        out.mkdir(parents=True, exist_ok=True)

        # Cap frame count (subsample evenly across captured epochs).
        max_frames = 100
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
            kept_frames = [frames[i] for i in idx]
        else:
            kept_frames = frames

        # Consistent y-axis limits across all frames
        all_vals = np.concatenate([f[1][:, :, 0].ravel() for f in kept_frames])
        obs_vals = np.asarray(test_data.x)
        if obs_vals.ndim == 2:
            obs_vals = obs_vals[:, 0]
        combined = np.concatenate([all_vals, obs_vals])
        y_lo, y_hi = float(combined.min()), float(combined.max())
        y_pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0
        y_lo -= y_pad
        y_hi += y_pad

        obs_t = np.asarray(test_data.t)

        # Fixed axis limits across all frames
        t_lo = float(obs_t.min())
        t_hi = float(obs_t.max())
        t_pad = 0.02 * (t_hi - t_lo) if t_hi > t_lo else 0.1
        t_lo -= t_pad
        t_hi += t_pad

        fig, ax = plt.subplots(figsize=(10, 5))

        def _draw(idx: int) -> None:
            ax.clear()
            epoch, traj, t_grid = kept_frames[idx]
            vals = traj[:, :, 0]  # (T, N)

            # Observed data
            ax.scatter(obs_t, obs_vals, s=3, c="grey", alpha=0.15, zorder=1)

            # Particle lines (subsample to 50)
            N = vals.shape[1]
            p_idx = np.linspace(0, N - 1, min(50, N), dtype=int)
            for i in p_idx:
                ax.plot(
                    t_grid,
                    vals[:, i],
                    lw=0.7,
                    alpha=0.4,
                    color="C0",
                    zorder=2,
                )
                ax.scatter(
                    t_grid,
                    vals[:, i],
                    s=8,
                    alpha=0.4,
                    color="C0",
                    zorder=3,
                )

            ax.set_xlim(t_lo, t_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title(f"epoch {epoch}")

        _draw(0)
        anim = animation.FuncAnimation(
            fig,
            typing.cast(Callable[[int], typing.Any], _draw),
            frames=len(kept_frames),
            interval=200,
            blit=False,
        )
        writer = animation.FFMpegWriter(fps=5, codec="h264")
        anim.save(str(out / "trajectory.mp4"), writer=writer)
        plt.close(fig)
        capped_note = (
            f" (capped from {len(frames)})" if len(kept_frames) < len(frames) else ""
        )
        print(
            f"  trajectory.mp4  "
            f"({len(kept_frames)} frames{capped_note}, "
            f"every {callback_every} epochs)"
        )

    return _callback, _finalize


# ------------------------------------------------------------------
# High-level convenience: produce all plots for a trained model
# ------------------------------------------------------------------


def plot_functional(
    potential_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    true_potential_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    interaction_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    true_interaction_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    test_data: SpatioTemporalData | None = None,
    save_path: str | Path | None = None,
    title: str = "Learned Functionals",
    n_grid: int = 200,
) -> Figure:
    """Plot learned (and optionally true) functionals: potential V(x) and
    interaction W(r).

    - 1-D data: line plots. Potential panel overlays true (solid) and learned
      (dashed) curves. If an interaction term is present, a second panel
      shows W(r) along the displacement axis.
    - 2-D data: heatmaps. Potential panel(s) show true / learned contours
      side-by-side (or just learned, if no ground truth). Interaction panel
      shows W(r) on a displacement grid centred at the origin.

    Each callable takes `(N, D)` points (with `D>=2` internally: 1-D fields
    ignore the second column) and returns `(N,)`.

    Potentials are shifted so their minimum is zero before plotting (they
    are defined up to a constant).
    """
    if test_data is not None:
        raw = np.asarray(test_data.x)
        dim = 1 if (raw.ndim == 1 or raw.shape[1] == 1) else raw.shape[1]
    else:
        dim = 2
        raw = None

    # Collect panels in order.
    panels: list[str] = []
    if potential_fn is not None or true_potential_fn is not None:
        panels.append("potential")
    if interaction_fn is not None or true_interaction_fn is not None:
        panels.append("interaction")
    if not panels:
        raise ValueError("plot_functional: nothing to plot (no functionals given).")

    if dim == 1:
        return _plot_functional_1d(
            potential_fn,
            true_potential_fn,
            interaction_fn,
            true_interaction_fn,
            raw,
            panels,
            save_path,
            title,
            n_grid,
        )
    return _plot_functional_2d(
        potential_fn,
        true_potential_fn,
        interaction_fn,
        true_interaction_fn,
        raw,
        panels,
        save_path,
        title,
        n_grid,
    )


def _plot_functional_1d(
    potential_fn,
    true_potential_fn,
    interaction_fn,
    true_interaction_fn,
    raw: np.ndarray | None,
    panels: list[str],
    save_path: str | Path | None,
    title: str,
    n_grid: int,
) -> Figure:
    # Axis range: data bounds with padding (fall back to [-10, 10]).
    if raw is not None and raw.size > 0:
        x_arr_raw = raw.ravel() if raw.ndim == 1 else raw[:, 0]
        x_min, x_max = float(x_arr_raw.min()), float(x_arr_raw.max())
        pad = 0.2 * max(x_max - x_min, 1e-3)
        x_min -= pad
        x_max += pad
    else:
        x_min, x_max = -10.0, 10.0
    x_arr = np.linspace(x_min, x_max, n_grid)
    pts = np.stack([x_arr, np.zeros_like(x_arr)], axis=-1).astype(np.float32)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(6 * len(panels), 4), squeeze=False
    )
    axes = axes.ravel()

    if "potential" in panels:
        ax = axes[panels.index("potential")]
        if true_potential_fn is not None:
            V_true = np.asarray(true_potential_fn(pts), dtype=float)
            V_true -= V_true.min()
            ax.plot(x_arr, V_true, color="steelblue", linewidth=2.0, label="true V(x)")
        if potential_fn is not None:
            V_learned = np.asarray(potential_fn(pts), dtype=float)
            V_learned -= V_learned.min()
            ax.plot(
                x_arr,
                V_learned,
                color="tomato",
                linewidth=2.0,
                linestyle="--",
                label="learned V(x)",
            )
        ax.set_xlabel("x")
        ax.set_ylabel("V(x)")
        ax.set_title("Potential")
        ax.legend()

    if "interaction" in panels:
        ax = axes[panels.index("interaction")]
        # Interaction over displacement r in [-span, span].
        span = x_max - x_min
        r_arr = np.linspace(-span, span, n_grid)
        r_pts = np.stack([r_arr, np.zeros_like(r_arr)], axis=-1).astype(np.float32)
        if true_interaction_fn is not None:
            W_true = np.asarray(true_interaction_fn(r_pts), dtype=float)
            ax.plot(r_arr, W_true, color="steelblue", linewidth=2.0, label="true W(r)")
        if interaction_fn is not None:
            W_learned = np.asarray(interaction_fn(r_pts), dtype=float)
            ax.plot(
                r_arr,
                W_learned,
                color="tomato",
                linewidth=2.0,
                linestyle="--",
                label="learned W(r)",
            )
        ax.set_xlabel("r")
        ax.set_ylabel("W(r)")
        ax.set_title("Interaction")
        ax.legend()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def _plot_functional_2d(
    potential_fn,
    true_potential_fn,
    interaction_fn,
    true_interaction_fn,
    raw: np.ndarray | None,
    panels: list[str],
    save_path: str | Path | None,
    title: str,
    n_grid: int,
) -> Figure:
    # Axis ranges.
    if raw is not None and raw.size > 0:
        flat = raw[:, :2]
        grid_range = float(max(np.abs(flat).max(), 1.0)) * 1.2
    else:
        grid_range = 5.0

    # How many subplot columns: potential gets 2 if both true+learned, else 1.
    pot_cols = int(true_potential_fn is not None) + int(potential_fn is not None)
    int_cols = int(true_interaction_fn is not None) + int(interaction_fn is not None)
    n_cols = max(pot_cols + int_cols, 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.2), squeeze=False)
    axes_flat = axes.ravel()

    gx = np.linspace(-grid_range, grid_range, n_grid)
    gy = np.linspace(-grid_range, grid_range, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32)

    col = 0
    # Potential panels
    for fn, label in [
        (true_potential_fn, "true V(x)"),
        (potential_fn, "learned V(x)"),
    ]:
        if fn is None:
            continue
        V = np.asarray(fn(pts), dtype=float).reshape(n_grid, n_grid)
        V -= V.min()
        ax = axes_flat[col]
        cf = ax.contourf(GX, GY, V, levels=30, cmap="RdYlBu_r")
        fig.colorbar(cf, ax=ax, shrink=0.8)
        ax.set_title(label)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_aspect("equal")
        col += 1

    # Interaction panels: W(r) on displacement grid
    if int_cols > 0:
        span = grid_range  # reuse same half-width for r
        gr = np.linspace(-span, span, n_grid)
        RX, RY = np.meshgrid(gr, gr)
        r_pts = np.stack([RX.ravel(), RY.ravel()], axis=-1).astype(np.float32)
        for fn, label in [
            (true_interaction_fn, "true W(r)"),
            (interaction_fn, "learned W(r)"),
        ]:
            if fn is None:
                continue
            W = np.asarray(fn(r_pts), dtype=float).reshape(n_grid, n_grid)
            ax = axes_flat[col]
            cf = ax.contourf(RX, RY, W, levels=30, cmap="RdYlBu_r")
            fig.colorbar(cf, ax=ax, shrink=0.8)
            ax.set_title(label)
            ax.set_xlabel("r₁")
            ax.set_ylabel("r₂")
            ax.set_aspect("equal")
            col += 1

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def save_results(
    samples_by_time: dict[float, jax.Array],
    test_data: SpatioTemporalData,
    potential_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    true_potential_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    interaction_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    true_interaction_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    model_trajectories: np.ndarray | None = None,
    model_t_grid: np.ndarray | None = None,
    output_dir: str | Path = "results",
    model_name: str = "model",
) -> Path:
    """Produce and save all visualisations for a training run.

    Creates:
      ``<output_dir>/<model_name>/snapshots.png``  — model density vs observed per-timepoint
      ``<output_dir>/<model_name>/functional.png`` — learned V(x) (+ W(r) if present),
        compared against true functionals where available
      ``<output_dir>/<model_name>/evolution.mp4``  — per-timepoint particle evolution
      ``<output_dir>/<model_name>/trajectory.png`` — (1-D) learned particle trajectories
      ``<output_dir>/<model_name>/trajectory.mp4`` — (1-D, from training callback)
        produced separately by :func:`make_trajectory_callback`.

    ``evolution.mp4`` animates draws from ``samples_by_time`` (the model
    density at each observation time), so every frame is a proper sample
    from p_θ(x, t), matching ``snapshots.png`` and EMD evaluation.

    ``model_trajectories`` / ``model_t_grid`` should be the learned
    particle paths with identity preserved across time (Stitching). When
    provided and the data is 1-D, ``trajectory.png`` is drawn from them so
    it matches the last frame of ``trajectory.mp4`` produced by
    :func:`make_trajectory_callback`. Models without particle identity
    across time (Lightspeed) omit ``trajectory.png``.
    """
    out = Path(output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)

    plot_snapshots(
        samples_by_time,
        test_data,
        potential_fn=potential_fn,
        save_path=out / "snapshots.png",
        title=f"{model_name} — Model vs Observed",
    )

    # --- Unified functional plot (V + optional W) --------------------------
    if potential_fn is not None or interaction_fn is not None:
        plot_functional(
            potential_fn=potential_fn,
            true_potential_fn=true_potential_fn,
            interaction_fn=interaction_fn,
            true_interaction_fn=true_interaction_fn,
            test_data=test_data,
            save_path=out / "functional.png",
            title=f"{model_name} — Learned Functionals",
        )

    # --- evolution.mp4: model *density* over time (KDE-sampled) -----------
    # Uses samples_by_time so the animation reflects the model density at
    # each timepoint, matching snapshots.png and EMD evaluation: for
    # Stitching these are KDE samples around the trajectory particles, so
    # each frame is a proper draw from p_θ(x, t), not raw particle centres.
    all_times = sorted(samples_by_time.keys())
    sample_t_grid = np.array(all_times)
    arrays = [np.asarray(samples_by_time[t]) for t in all_times]
    max_n = max(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[0] < max_n:
            pad_rows = np.repeat(a[-1:], max_n - a.shape[0], axis=0)
            a = np.concatenate([a, pad_rows], axis=0)
        padded.append(a)
    sampled_trajectories = np.stack(padded)  # (T, N, D)

    animate_particles(
        trajectories=sampled_trajectories,
        t_grid=sample_t_grid,
        test_data=test_data,
        potential_fn=potential_fn,
        save_path=out / "evolution.mp4",
        title=f"{model_name} — Model Density Evolution",
    )

    # --- trajectory.png: learned particle *paths* with identity -----------
    # Only meaningful for models that carry trajectories (Stitching).
    # This is the same source used by trajectory.mp4, so the png matches
    # the mp4's last frame.
    dim = sampled_trajectories.shape[2]
    if dim == 1 and model_trajectories is not None and model_t_grid is not None:
        real_traj = np.asarray(model_trajectories)
        if real_traj.ndim == 2:
            real_traj = real_traj[..., None]
        plot_trajectory(
            trajectories=real_traj,
            t_grid=np.asarray(model_t_grid),
            test_data=test_data,
            save_path=out / "trajectory.png",
            title=f"{model_name} — Trajectories",
        )

    return out


def visualise(
    model,
    cfg,
    train_data: SpatioTemporalData,
    test_data: SpatioTemporalData,
    output_dir: str = "results",
    true_potential_fn=None,
):
    """Generate scatter plots and particle-evolution animation.

    Both `snapshots.png` and `evolution.mp4` are produced from the same
    `samples_by_time` dict sampled at the unique observation timepoints,
    so they always cover the same time span with the same frames.

    When *true_potential_fn* is provided (synthetic datasets), an additional
    `potential_comparison.png` is saved comparing the analytic and learned
    potentials.

    Returns the output directory path.
    """
    import jax
    import jax.numpy as jnp

    from stitching.build import spec_for_model

    t_unique = jnp.unique(test_data.t)

    viz_key = jax.random.key(cfg.seed + 1)
    if spec_for_model(model).needs_train_data_for_sample:
        samples_by_time = model.sample(t_unique, train_data)
    else:
        # Stitching: the model density is a KDE around the trajectory
        # particles (same representation used in training loss and EMD
        # evaluation). Draw plenty of samples from that KDE so the
        # histogram is smooth rather than a forest of Dirac spikes at the
        # particle centres. We oversample (default 10k) because
        # visualisation is cheap and histogram bin noise scales as
        # 1/sqrt(N_per_bin).
        n = cfg.eval_num_samples or 10000
        samples_by_time = model.sample(t_unique, num_samples=n, key=viz_key)

    dim = train_data.x.shape[1]

    potential_fn: Callable[[np.ndarray], np.ndarray] | None = None
    c_pot = float(jax.nn.softplus(model.raw_potential_coeff))

    def _potential_fn(pts_2d: np.ndarray) -> np.ndarray:
        pts = np.zeros((pts_2d.shape[0], dim), dtype=np.float32)
        pts[:, : min(2, dim)] = pts_2d[:, : min(2, dim)]
        vals = jax.vmap(model.learned_potential_fn())(jnp.array(pts))
        return np.asarray(c_pot * vals)

    potential_fn = _potential_fn

    # --- Interaction function (stitching models with interaction_net) ----
    interaction_fn: Callable[[np.ndarray], np.ndarray] | None = None
    from stitching.models import Stitching

    if isinstance(model, Stitching) and model.interaction_net is not None:
        c_inter = float(jax.nn.softplus(model.raw_interaction_coeff))
        inter_net = model.interaction_net
        inter_type = model.interaction_type

        def _interaction_fn(r_2d: np.ndarray) -> np.ndarray:
            r = np.zeros((r_2d.shape[0], dim), dtype=np.float32)
            r[:, : min(2, dim)] = r_2d[:, : min(2, dim)]
            r_jax = jnp.array(r)
            if inter_type == "radial":
                vals = jax.vmap(
                    lambda ri: jnp.squeeze(inter_net(jnp.sum(ri**2, keepdims=True)))
                )(r_jax)
            else:
                vals = jax.vmap(lambda ri: jnp.squeeze(inter_net(ri)))(r_jax)
            return np.asarray(c_inter * vals)

        interaction_fn = _interaction_fn

    # --- Real learned particle trajectories (if available) ----------------
    model_trajectories: np.ndarray | None = None
    model_t_grid: np.ndarray | None = None
    if hasattr(model, "trajectories") and hasattr(model, "t_grid"):
        model_trajectories = np.asarray(model.trajectories)
        model_t_grid = np.asarray(model.t_grid)

    return save_results(
        samples_by_time=samples_by_time,
        test_data=test_data,
        potential_fn=potential_fn,
        true_potential_fn=true_potential_fn,
        interaction_fn=interaction_fn,
        model_trajectories=model_trajectories,
        model_t_grid=model_t_grid,
        output_dir=output_dir,
        model_name=f"{cfg.experiment_name}-{cfg.model}",
    )

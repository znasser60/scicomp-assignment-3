"""Animations of eigenmode solutions to 2D wave equation on a circular domain."""

import argparse
import math
import typing
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.axes3d import Axes3D

from scicomp.eig_val_calc.equation_solver_components.solving_equation import (
    eval_oscillating_solution,
)
from scicomp.eig_val_calc.solvers import solve_circle_laplacian


def animate_eigenmode(
    mode: npt.NDArray[np.float64],
    freq: float,
    length: float,
    n: int,
    c: float,
    index_grid: npt.NDArray[np.float64],
    animation_speed: float = 1.0,
    repeats: int = 1,
    fps: int = 60,
) -> FuncAnimation:
    """Create an animation of eigenmode solution to the 2D wave equation on a circle.

    Args:
        mode: Eigenmode as a 1D Numpy array of the discrete cells which lie within
            the circle.
        freq: Corresponding eigenfrequency, defined as freq = sqrt(-lambda)
        length: Physical diameter of the circle (m).
        n: Number of intervals used to divide the cartesian axes.
        c: Wave propagation velocity (m/s)
        index_grid: Matrix of indices for cells corresponding to points within the
            circle. Points outside the circle are assigned None.
        animation_speed: Speed at which to display animation, default is 1 T/s where
            T is the period of oscillation.
        repeats: Number of complete oscillations to show in animation.
        fps: Framerate of animation.

    Returns:
        Matplotlib FuncAnimation object.
    """
    solve = partial(eval_oscillating_solution, mode=mode, freq=freq, c=c)

    period = 2 * np.pi / (freq * c)
    runtime = (period * repeats) / animation_speed
    n_frames = int(math.ceil(runtime * fps))
    frame_times = np.linspace(0, period, n_frames)

    # Prepare coordinates for drum
    x = np.linspace(-length / 2, length / 2, n)
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f"$\\lambda={freq:.2f}$hz, ({animation_speed}x speed)")
    ax = fig.add_subplot(111, projection="3d")

    # Satisfy type-checker that this is indeed a 3D axis
    ax = typing.cast(Axes3D, ax)

    max_abs_v = max(np.abs(mode))
    min_val = -1.5 * max_abs_v
    max_val = 1.5 * max_abs_v

    ax.set_zlim(2 * min_val, 2 * max_val)
    amplitude = np.full((n, n), np.nan)
    amplitude[~np.isnan(index_grid)] = solve(0)
    ax.plot_surface(X, Y, amplitude, cmap="coolwarm", vmin=min_val, vmax=max_val)

    def update(t):
        ax.clear()
        ax.set_zlim(2 * min_val, 2 * max_val)
        amplitude[~np.isnan(index_grid)] = solve(t)
        surface = ax.plot_surface(
            X, Y, amplitude, cmap="coolwarm", vmin=min_val, vmax=max_val
        )
        return [surface]

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_times,
        interval=1000 / fps,
        repeat=False,
    )

    return ani


# def main(
#    k: int,
#    n: int,
#    animation_speed: float,
#    repeats: int,
#    fps: int,
#    quality_label: str
# ):
#    """Create animations."""
#    length = 1.0
#    c = 1.0
#    n = 500
#    ks = np.concatenate([np.arange(10), np.array([38, 50, 51])], axis=0)
#    ks = np.arange(10)
#    fps = 10
#    use_sparse = True
#    shift_invert = True
#
#    eigenfrequencies, eigenmodes, index_grid = solve_circle_laplacian(
#        length=length,
#        n=n,
#        k=max(ks) + 1,
#        use_sparse=use_sparse,
#        shift_invert=shift_invert,
#    )
#
#    for k in ks:
#        ani = animate_eigenmode(
#            eigenmodes[:, k],
#            eigenfrequencies[k].item(),
#            length=length,
#            n=n,
#            c=c,
#            index_grid=index_grid,
#            animation_speed=animation_speed,
#            repeats=repeats,
#            fps=fps,
#        )
#        ani.save(f"circular_drum_k_{k}_{quality_label}_quality.mp4", dpi=100)


def main(
    k: int,
    n: int,
    animation_speed: float,
    repeats: int,
    fps: int,
    dpi: int,
    quality_label: str,
):
    """Create animations."""
    length = 1.0
    c = 1.0
    use_sparse = True
    shift_invert = True

    eigenfrequencies, eigenmodes, index_grid = solve_circle_laplacian(
        length=length,
        n=n,
        k=k + 1,
        use_sparse=use_sparse,
        shift_invert=shift_invert,
    )

    ani = animate_eigenmode(
        eigenmodes[:, k],
        eigenfrequencies[k].item(),
        length=length,
        n=n,
        c=c,
        index_grid=index_grid,
        animation_speed=animation_speed,
        repeats=repeats,
        fps=fps,
    )
    ani.save(
        f"results/animations/circular_drum_k_{k}_{quality_label}_quality.mp4",
        dpi=dpi,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--animation-speed", type=float)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--quality-label", type=str)
    args = parser.parse_args()
    main(
        k=args.k,
        n=args.n,
        animation_speed=args.animation_speed,
        repeats=args.repeats,
        fps=args.fps,
        dpi=args.dpi,
        quality_label=args.quality_label,
    )

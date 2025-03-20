"""Animations of eigenmode solutions to 2D wave equation on a circular domain."""

import math
import typing
from functools import partial
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import typer
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.axes3d import Axes3D

from scicomp.domains import ShapeEnum
from scicomp.eig_val_calc.equation_solver_components.solving_equation import (
    eval_oscillating_solution,
)

app = typer.Typer()


@app.command()
def eigenmode(
    shape: Annotated[ShapeEnum, typer.Option("--domain", help="Shape type.")],
    k: Annotated[int, typer.Option("--k", help="Number of eigenvalues to compute.")],
    n: Annotated[
        int,
        typer.Option(
            "--n", help="Number of intervals used to divide the cartesian axes."
        ),
    ],
    animation_speed: Annotated[
        float,
        typer.Option(
            "--animation-speed",
            help="Speed at which to display animation, default is 1 T/s where T is the "
            "period of oscillation.",
        ),
    ],
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats", help="Number of complete oscillations to show in animation."
        ),
    ],
    fps: Annotated[int, typer.Option("--fps", help="Framerate of animation.")],
    dpi: Annotated[int, typer.Option("--dpi", help="DPI of animation.")],
    quality_label: Annotated[
        str,
        typer.Option(
            "--quality-label",
            help="The quality of the plot, as specified in the file name.",
        ),
    ],
    width: Annotated[int, typer.Option("--width", help="Physical width of shape.")] = 1,
    height: Annotated[
        int | None,
        typer.Option(
            "--height",
            help="Physical height of shape (only applicable for rectangles).",
        ),
    ] = None,
):
    """Create animations."""
    c = 1.0
    use_sparse = True
    shift_invert = False

    domain = shape.domain(width, height)
    index_grid = domain.discretise(n)
    eigenfrequencies, eigenmodes = domain.solve_eigenproblem(
        k=k + 1,
        ny=n,
        index_grid=index_grid,
        use_sparse=use_sparse,
        shift_invert=shift_invert,
    )

    ani = animate_eigenmode(
        eigenmodes[:, k],
        eigenfrequencies[k].item(),
        length=width,
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
    x = np.linspace(-length / 2, length / 2, n + 1)
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
    amplitude = np.full((n + 1, n + 1), np.nan)
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

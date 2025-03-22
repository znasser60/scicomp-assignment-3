from typing import Annotated

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import typer
from matplotlib.axes import Axes

from scicomp.domains import ShapeEnum

app = typer.Typer()


@app.command()
def circular_steady_state_diffusion(
    length: Annotated[
        int,
        typer.Option("--length", help="Diameter of the circular domain."),
    ] = 4,
    n: Annotated[
        int, typer.Option("--n", help="Number of grid points in each dimension.")
    ] = 150,
    source_position: Annotated[
        tuple[int, int],
        typer.Option("--source-position", help="Position of the source on the grid."),
    ] = (0.6, 1.2),
    quality_label: Annotated[
        str,
        typer.Option(
            "--quality-label",
            help="The quality of the plot, as specified in the file name.",
        ),
    ] = "undefined",
):
    """Plots the steady-state diffusion solution on a circular domain."""
    shape = ShapeEnum.Circle
    domain = shape.domain(width=length)

    c_grid = domain.solve_diffusion(source_position, n, use_sparse=True)

    radius = length / 2
    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
    heatmap = plot_circle_diffusion(n, c_grid, radius, ax=ax)
    fig.colorbar(heatmap, ax=ax, label="c(x, y)")

    # Tidy up plot
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-radius * 1.02, radius * 1.02)
    ax.set_ylim(-radius * 1.02, radius * 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.savefig(
        f"results/figures/steady_state_diffusion_{quality_label}_quality.png",
        bbox_inches="tight",
    )
    fig.savefig(
        f"results/figures/steady_state_diffusion_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )


def plot_circle_diffusion(
    n: int, c_grid: npt.NDArray[np.float64], radius: float, ax: Axes | None = None
):
    """Plots the steady-state concentration solution on the circular domain."""
    if ax is None:
        ax = plt.gca()

    x = np.linspace(-radius, radius, n + 1)
    y = np.linspace(-radius, radius, n + 1)
    X, Y = np.meshgrid(x, y)
    mask = (radius) ** 2 > (X**2 + Y**2)

    vmin = c_grid[mask].min()
    heatmap = ax.imshow(
        c_grid[::-1],
        norm=colors.LogNorm(vmin=vmin, vmax=1),
        cmap="magma",
        extent=(-radius, radius, -radius, radius),
    )
    return heatmap


circular_steady_state_diffusion(4, 150)
alma = 1

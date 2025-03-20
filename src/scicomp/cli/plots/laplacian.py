"""Plot the discrete Laplacian for a given shape and discretisation."""

from typing import Annotated

import matplotlib.pyplot as plt
import typer

from scicomp.domains import ShapeEnum

app = typer.Typer()


@app.command(name="laplacian")
def plot_laplacian(
    shape: Annotated[
        ShapeEnum,
        typer.Option("--domain", help="Shape type."),
    ],
    n: Annotated[
        int,
        typer.Option(
            "--n", help="Number of intervals used to divide the cartesian axes."
        ),
    ],
    width: Annotated[
        int,
        typer.Option("--width", help="Physical width of shape."),
    ] = 1,
    height: Annotated[
        int | None,
        typer.Option(
            "--height",
            help="Physical height of shape (only applicable for rectangles).",
        ),
    ] = None,
):
    """Plot the discrete Laplacian for a shape and discretisation."""
    domain = shape.domain(width, height)
    laplacian = domain.construct_discrete_laplacian(n)

    _, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
    ax.imshow(laplacian)

    plt.show()

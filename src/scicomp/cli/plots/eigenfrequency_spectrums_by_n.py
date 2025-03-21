"""Plot spectrum of eigenfrequencies for varying N."""

from typing import Annotated

import matplotlib.pyplot as plt
import typer

from scicomp.utils.plot import plot_eigenspectrum_by_n

app = typer.Typer()


@app.command()
def eigenspectrum_by_n(
    min_n: Annotated[
        int,
        typer.Option(
            "--min-n",
        ),
    ],
    max_n: Annotated[
        int,
        typer.Option(
            "--max-n",
        ),
    ],
):
    """Plot spectrum of eigenfrequencies for varying N."""
    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)

    plot_eigenspectrum_by_n(min_n, max_n, ax=ax)
    fig.savefig(
        "results/figures/eigenfrequency_spectrum_by_n.pdf",
        bbox_inches="tight",
    )

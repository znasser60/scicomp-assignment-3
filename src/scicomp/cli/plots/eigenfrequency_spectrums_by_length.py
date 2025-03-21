"""Plot spectrum of eigenfrequencies for varying N."""

from typing import Annotated

import matplotlib.pyplot as plt
import typer

from scicomp.utils.plot import plot_eigenspectrum_by_length

app = typer.Typer()


@app.command()
def eigenspectrum_by_length(
    n_at_unit_length: Annotated[
        int,
        typer.Option(
            "--n-at-unit-length",
        ),
    ],
    quality_label: Annotated[
        str,
        typer.Option(
            "--quality-label",
        ),
    ] = "undefined",
):
    """Plot spectrum of eigenfrequencies for varying length."""
    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
    plot_eigenspectrum_by_length(n_at_unit_length, ax=ax)
    fig.savefig(
        f"results/figures/eigenfrequency_spectrum_by_length_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )

"""Plot spectrum of eigenfrequencies for varying N."""

from fractions import Fraction
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer

from scicomp.domains import Circle

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
    h = Fraction(1, n_at_unit_length)
    k = n_at_unit_length - 1

    lengths = [Fraction(length, 10) for length in np.arange(10, 51, 5)]

    data = {
        "length": [],
        "lambda": [],
    }
    for length in lengths:
        domain = Circle(length)
        n = (length / h).numerator
        eigenfrequencies, _ = domain.solve_eigenproblem(
            k, n, use_sparse=True, shift_invert=True
        )
        data["length"].extend(np.repeat(float(length), len(eigenfrequencies)).tolist())
        data["lambda"].extend(eigenfrequencies.tolist())

    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
    sns.boxplot(
        x=data["length"],
        y=data["lambda"],
        fill=False,
        color="grey",
        width=0.2,
        native_scale=True,
        ax=ax,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(0, None)

    ax.set_xlabel("Object length ($m$)")
    ax.set_ylabel("$\\lambda$ (hz)")

    fig.savefig(
        f"results/figures/eigenfrequency_spectrum_by_length_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )

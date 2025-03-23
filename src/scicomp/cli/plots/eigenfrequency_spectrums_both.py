"""Plot spectrum of eigenfrequencies for varying N and L."""

from typing import Annotated

import matplotlib.pyplot as plt
import seaborn as sns
import typer

from scicomp.model_elements.domains import ShapeEnum
from scicomp.utils.plot import plot_eigenspectrum_by_length, plot_eigenspectrum_by_n

app = typer.Typer()


@app.command()
def eigenspectrums(
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
    quality_label: Annotated[
        str,
        typer.Option(
            "--quality-label",
        ),
    ] = "undefined",
):
    """Plot eigenfrequency spectrums for varying N and L."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.3, 2),
        sharey=True,
        constrained_layout=True,
        gridspec_kw=dict(width_ratios=[0.6, 1.0]),
    )

    palette = sns.color_palette()
    plot_eigenspectrum_by_n(min_n, max_n, palette=palette[:2], ax=axes[0])
    plot_eigenspectrum_by_length(
        max_n, palette=palette[2 : 2 + len(ShapeEnum)], ax=axes[1]
    )

    for label in axes[0].get_xmajorticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    axes[1].tick_params(axis="y", left=False)

    fig.savefig(
        f"results/figures/eigenfrequency_spectrums_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )

"""Plot first few eigenmodes for each shape."""

import matplotlib.pyplot as plt
import typer

from scicomp.domains import Circle, Rectangle
from scicomp.utils.plot import plot_shape_eigenmodes

app = typer.Typer()


@app.command()
def eigenmodes():
    """Plot first k eigenmodes (columns) for each shape (rows)."""
    length = 1
    n = 500
    k = 5
    use_sparse = True
    shift_invert = True

    fig, axes = plt.subplots(
        3,
        k,
        figsize=(3.3, 2.25),
        sharex="row",
        sharey="row",
        constrained_layout=True,
        gridspec_kw=dict(height_ratios=[1, 1, 0.8]),
    )

    for ax in axes[:, k - 1]:
        ax.yaxis.set_label_position("right")

    plot_shape_eigenmodes(
        domain=Circle(length),
        n=n,
        use_sparse=use_sparse,
        shift_invert=shift_invert,
        axes=axes[0],
    )
    axes[0, k - 1].set_ylabel("Circle", rotation=0, labelpad=10, ha="left", va="center")

    plot_shape_eigenmodes(
        domain=Rectangle(length),
        n=n,
        use_sparse=use_sparse,
        shift_invert=shift_invert,
        axes=axes[1],
    )
    axes[1, k - 1].set_ylabel("Square", rotation=0, labelpad=10, ha="left", va="center")

    plot_shape_eigenmodes(
        domain=Rectangle(length * 2, length),
        n=n,
        use_sparse=use_sparse,
        shift_invert=shift_invert,
        axes=axes[2],
    )
    axes[2, k - 1].set_ylabel("Rect.", rotation=0, labelpad=10, ha="left", va="center")

    fig.savefig("results/figures/eigenmodes.pdf", bbox_inches="tight")

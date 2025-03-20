"""Plot first few eigenmodes for each shape."""

import matplotlib.pyplot as plt
import typer

from scicomp.domains import Circle, Rectangle
from scicomp.utils.plot import plot_eigenmode

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
        3, k, figsize=(3.6, 2.5), sharex="row", sharey="row", constrained_layout=True
    )

    for ax in axes[:, k - 1]:
        ax.yaxis.set_label_position("right")

    axes[0, k - 1].set_ylabel("Rect.", rotation=0, labelpad=10, ha="left", va="center")
    domain = Rectangle(length * 2, length)
    index_grid = domain.discretise(n)
    eigenfrequencies, eigenmodes = domain.solve_eigenproblem(
        k=k,
        ny=n,
        use_sparse=use_sparse,
        shift_invert=shift_invert,
        index_grid=index_grid,
    )
    for i, ax in enumerate(axes[0]):
        plot_eigenmode(
            eigenmodes[:, i],
            eigenfrequencies[i],
            float(domain.width),
            float(domain.height),
            index_grid,
            ax=ax,
        )
        ax.set_title(f"$\\omega = {eigenfrequencies[i]:.2f}$")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[1, k - 1].set_ylabel("Square", rotation=0, labelpad=10, ha="left", va="center")
    domain = Rectangle(length)
    index_grid = domain.discretise(n)
    eigenfrequencies, eigenmodes = domain.solve_eigenproblem(
        k=k,
        ny=n,
        use_sparse=use_sparse,
        shift_invert=shift_invert,
        index_grid=index_grid,
    )
    for i, ax in enumerate(axes[1]):
        plot_eigenmode(
            eigenmodes[:, i],
            eigenfrequencies[i],
            float(domain.width),
            float(domain.height),
            index_grid,
            ax=ax,
        )
        ax.set_title(f"$\\omega = {eigenfrequencies[i]:.2f}$")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    domain = Circle(length)
    index_grid = domain.discretise(n)
    eigenfrequencies, eigenmodes = domain.solve_eigenproblem(
        k=k,
        ny=n,
        use_sparse=use_sparse,
        shift_invert=shift_invert,
        index_grid=index_grid,
    )
    axes[2, k - 1].set_ylabel("Circle", rotation=0, labelpad=10, ha="left", va="center")
    for i, ax in enumerate(axes[2]):
        plot_eigenmode(
            eigenmodes[:, i],
            eigenfrequencies[i],
            float(domain.width),
            float(domain.height),
            index_grid,
            ax=ax,
        )
        ax.set_title(f"$\\omega = {eigenfrequencies[i]:.2f}$")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig("results/figures/eigenmodes.pdf", bbox_inches="tight")

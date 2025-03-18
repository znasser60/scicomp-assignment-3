"""Plot first few eigenmodes for each shape."""

import matplotlib.pyplot as plt

from scicomp.domains import Circle
from scicomp.utils.plot import plot_eigenmode


def main():
    """Plot first k eigenmodes (columns) for each shape (rows)."""
    length = 1
    n = 200
    k = 4

    fig, axes = plt.subplots(
        3, k, figsize=(3, 3), sharex="row", sharey="row", constrained_layout=True
    )

    for ax in axes[:, k - 1]:
        ax.yaxis.set_label_position("right")

    # TODO: Add square and rectangle
    axes[0, k - 1].set_ylabel("Square", rotation=0, labelpad=10, ha="left", va="center")
    axes[1, k - 1].set_ylabel("Rect.", rotation=0, labelpad=10, ha="left", va="center")

    domain = Circle(length)
    index_grid = domain.discretise(n)
    eigenfrequencies, eigenmodes = domain.solve_eigenproblem(
        k=k,
        use_sparse=True,
        shift_invert=True,
        index_grid=index_grid,
    )
    axes[2, k - 1].set_ylabel("Circle", rotation=0, labelpad=10, ha="left", va="center")
    for i, ax in enumerate(axes[2]):
        plot_eigenmode(
            eigenmodes[:, i], eigenfrequencies[i], length, n, index_grid, ax=ax
        )
        ax.set_title(f"$\\lambda = {eigenfrequencies[i]:.2f}$")
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

    fig.savefig("eigenmodes.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

"""Plot first few eigenmodes for each shape."""

import matplotlib.pyplot as plt

from scicomp.eig_val_calc.circle import plot_eigenmode, solve_circle_laplacian


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

    eigenfrequencies, eigenmodes, index_grid = solve_circle_laplacian(
        length, n, k, use_sparse=True, shift_invert=True
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

    fig.savefig("results/figures/eigenmodes.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

"""Plot spectrum of eigenfrequencies for varying N."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from scicomp.eig_val_calc.circle import solve_circle_laplacian


def main(max_n: int, quality_label: str):
    """Plot spectrum of eigenfrequencies for varying N."""
    length = 1
    min_n = 50

    n_rows = (((max_n + 1) - min_n) // 20) + 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(1.5, 4), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, n in zip(axes.flatten(), np.arange(50, max_n + 1, 20), strict=True):
        k = n - 1
        eigenfrequencies, *_ = solve_circle_laplacian(
            length, n, k, use_sparse=True, shift_invert=True
        )
        ax.hist(eigenfrequencies)
        ax.set_ylabel(f"$N={n}$")

    axes[0].set_xlim(0, None)

    fig.savefig(
        f"results/figures/eigenfrequency_spectrum_by_n_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int)
    parser.add_argument("--quality-label", type=str)
    args = parser.parse_args()
    main(args.max_n, args.quality_label)

"""Plot spectrum of eigenfrequencies."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from scicomp.eig_val_calc.circle import solve_circle_laplacian


def main(n, quality_label):
    """Plot spectrum of eigenfrequencies for varying shape size L."""
    k = n - 1

    fig, axes = plt.subplots(
        6, 1, figsize=(1.5, 4), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, length in zip(axes.flatten(), np.logspace(-2, 3, 6), strict=True):
        eigenfrequencies, *_ = solve_circle_laplacian(
            length, n, k, use_sparse=True, shift_invert=True
        )
        ax.hist(eigenfrequencies)
        ax.set_ylabel(f"$L={length}$")

    fig.savefig(
        f"results/figures/eigenfrequency_spectrum_by_length_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--quality-label", type=str)
    args = parser.parse_args()
    main(args.n, args.quality_label)

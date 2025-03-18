"""Measure difference in results between sparse and dense eigensolvers."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from scicomp.eig_val_calc.circle import (
    construct_circle_laplacian,
    initialize_grid,
    solve_circle_laplacian,
)


def main(max_n: int, quality_label: str):
    """Create plot of differences in results between sparse and dense eigensolvers.

    For varying values of N, the number of intervals used to divide the cartesian
    axes, we compute the eigenvalues of the circle Laplacian using the following
    solvers/configurations:

    - Dense solver (scipy.linalg.eigh)
    - Sparse solver (scipy.linalg.sparse.eigsh)
    - Sparse solver with shift-invert

    We record the eigenvalue corresponding to the dominant eigenfrequency (the smallest
    magnitude eigenvalue) for each method, and plot the absolute and relative
    differences between the sparse methods and the dense method.
    """
    length = 1
    k = 5

    dom_dense = []
    dom_sparse = []
    dom_sparse_si = []
    ns = np.arange(5, max_n, 5)
    for n in tqdm(ns, desc="Compare eigensolver results"):
        _, index_grid = initialize_grid(length, n)
        dense_laplacian = construct_circle_laplacian(
            index_grid, length, n, use_sparse=False
        )
        freqs, _ = solve_circle_laplacian(dense_laplacian, n, k)
        dom_dense.append(freqs[0])

        sparse_laplacian = construct_circle_laplacian(
            index_grid, length, n, use_sparse=True
        )
        freqs, _ = solve_circle_laplacian(sparse_laplacian, n, k)
        dom_sparse.append(freqs[0])
        freqs, _ = solve_circle_laplacian(sparse_laplacian, n, k, shift_invert=True)
        dom_sparse_si.append(freqs[0])

    dom_dense = np.asarray(dom_dense)
    dom_sparse = np.asarray(dom_sparse)
    dom_sparse_si = np.asarray(dom_sparse_si)

    abs_diff_sparse_dense = np.abs(dom_sparse - dom_dense)
    abs_diff_sparse_si_dense = np.abs(dom_sparse_si - dom_dense)

    rel_diff_sparse_dense = abs_diff_sparse_dense / dom_dense
    rel_diff_sparse_si_dense = abs_diff_sparse_si_dense / dom_dense

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.5), constrained_layout=True)
    axes[0].plot(ns, abs_diff_sparse_dense)
    axes[0].plot(ns, abs_diff_sparse_si_dense)
    axes[1].plot(ns, rel_diff_sparse_dense)
    axes[1].plot(ns, rel_diff_sparse_si_dense)

    axes[0].set_ylabel("Absolute error")
    axes[1].set_ylabel("Relative error")

    for ax in axes:
        ax.set_xlabel("N")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(
        f"results/figures/compare_results_eigensolvers_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int)
    parser.add_argument("--quality-label", type=str)
    args = parser.parse_args()
    main(args.max_n, args.quality_label)

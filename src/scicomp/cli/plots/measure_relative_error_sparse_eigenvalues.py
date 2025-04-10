"""Measure difference in results between sparse and dense eigensolvers."""

from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from tqdm import tqdm

from scicomp.model_elements.domains import Circle

app = typer.Typer()


@app.command()
def compare_eigensolver_results(
    max_n: Annotated[int, typer.Option("--max-n")],
    quality_label: Annotated[
        str,
        typer.Option(
            "--quality-label",
            help="The quality of the plot, as specified in the file name.",
        ),
    ] = "undefined",
):
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

    domain = Circle(length)

    dom_dense = []
    dom_sparse = []
    dom_sparse_si = []
    ns = np.arange(5, max_n, 5)
    for n in tqdm(ns, desc="Compare eigensolver results"):
        index_grid = domain.discretise(n)
        dense_laplacian = domain.construct_discrete_laplacian(
            use_sparse=False, index_grid=index_grid
        )
        sparse_laplacian = domain.construct_discrete_laplacian(
            use_sparse=True, index_grid=index_grid
        )

        freqs, _ = domain.solve_eigenproblem(
            k=k, ny=n, laplacian=dense_laplacian, index_grid=index_grid
        )
        dom_dense.append(freqs[0])

        freqs, _ = domain.solve_eigenproblem(
            k=k,
            ny=n,
            laplacian=sparse_laplacian,
            index_grid=index_grid,
            shift_invert=False,
        )
        dom_sparse.append(freqs[0])

        freqs, _ = domain.solve_eigenproblem(
            k=k,
            ny=n,
            laplacian=sparse_laplacian,
            index_grid=index_grid,
            shift_invert=True,
        )
        dom_sparse_si.append(freqs[0])

    dom_dense = np.asarray(dom_dense)
    dom_sparse = np.asarray(dom_sparse)
    dom_sparse_si = np.asarray(dom_sparse_si)

    abs_diff_sparse_dense = np.abs(dom_sparse - dom_dense)
    abs_diff_sparse_si_dense = np.abs(dom_sparse_si - dom_dense)

    rel_diff_sparse_dense = abs_diff_sparse_dense / dom_dense
    rel_diff_sparse_si_dense = abs_diff_sparse_si_dense / dom_dense

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.35), constrained_layout=True)
    axes[0].plot(ns, abs_diff_sparse_dense)
    axes[0].plot(ns, abs_diff_sparse_si_dense)
    axes[1].plot(ns, rel_diff_sparse_dense)
    axes[1].plot(ns, rel_diff_sparse_si_dense)

    axes[0].set_ylabel("Abs. error")
    axes[1].set_ylabel("Rel. error")

    for ax in axes:
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supxlabel("$N$")

    fig.savefig(
        f"results/figures/compare_results_eigensolvers_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )

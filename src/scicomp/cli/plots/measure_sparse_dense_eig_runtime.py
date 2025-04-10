"""Compare performance of different eigensolvers."""

from time import perf_counter
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import typer
from matplotlib.axes import Axes
from tqdm import tqdm

from scicomp.model_elements.domains import Circle

"""Z-score for 2 standard deviations."""
Z2STDEV = 1.97

app = typer.Typer()


@app.command()
def compare_eigensolver_runtime(
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats",
            help="Number of repeats used to calculate the sample standard deviation.",
        ),
    ],
    timeout: Annotated[
        float, typer.Option("--timeout", help="Maximum runtime before termination.")
    ],
    quality_label: Annotated[
        str,
        typer.Option(
            "--quality-label",
            help="The quality of the plot, as specified in the file name.",
        ),
    ] = "undefined",
):
    """Create and save plot comparing runtime of different eigensolvers."""
    fig, ax = plt.subplots(figsize=(3.3, 1.5), layout="constrained")
    compare_runtime_sparse_vs_dense(
        length=1, ns=np.arange(10, 300, 10), repeats=repeats, timeout=timeout, ax=ax
    )
    ax.set_ylim(0, 4)
    fig.savefig(
        f"results/figures/compare_runtime_eigensolvers_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )


def measure_runtime(
    length,
    ns,
    repeats: int = 1,
    use_sparse: bool = False,
    shift_invert: bool = False,
    use_eigsh: bool = False,
    timeout: float | None = None,
    prog_bar: tqdm | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Measure the performance of an eigensolver for different discretisations.

    Varies N, the number of spatial intervals used to divide the cartesian axes,
    in a given range, measuring the average runtime. This is repeated a number of
    times to obtain an estimate of the standard deviation in runtimes.

    If a timeout is specified, then after all repeats for a given N, the timeout
    is compared to the lower end of the confidence interval for that N. If the
    lower confidence bound exceeds the timeout, then the mean runtime for that N
    exceeds the timeout with 95% confidence, and we terminate the experiment,
    returning None for any subsequent values of N.

    Args:
        length: Physical side length of the object.
        ns: Numpy array of values to test for N, the number of spatial intervals
            used to divide the cartesian axes.
        repeats: Number of repeats used to calculate the sample standard deviation.
        use_sparse: Use a sparse eigensolver.
        use_eigsh: Use the eigsh eigensolver (applicable for sparse eigensolvers).
        shift_invert: Use the shift-invert setting in eigensolver (only applicable
            for sparse eigensolvers).
        timeout: Maximum runtime before termination.
        prog_bar: Optional progress bar. If provided, function updates the progress
            bar after each run.

    Returns:
        Tuple contaning two 1D Numpy arrays: (mean, std), each with length equal
        to the number of elements specified in the `ns` parameter.
    """
    runtime = np.empty((repeats, len(ns)), dtype=np.float64)

    domain = Circle(length)
    for n_i, n in enumerate(ns):
        if prog_bar is not None:
            if use_sparse and shift_invert:
                mode = "sparse + shift-inv"
            elif use_sparse and use_eigsh:
                mode = "sparse (eigsh)"
            elif use_sparse:
                mode = "sparse (eigs)"
            else:
                mode = "dense"
            prog_bar.set_description(f"Measure eig-solver runtime ({mode=}, N={n})")

        index_grid = domain.discretise(n)
        laplacian = domain.construct_discrete_laplacian(
            use_sparse=use_sparse, index_grid=index_grid
        )
        for r in range(repeats):
            start = perf_counter()
            domain.solve_eigenproblem(
                k=n - 1,
                ny=n,
                index_grid=index_grid,
                laplacian=laplacian,
                shift_invert=shift_invert,
                use_eigsh=use_eigsh,
            )
            duration = perf_counter() - start
            runtime[r, n_i] = duration
            if prog_bar is not None:
                prog_bar.update()
        if timeout is not None:
            mean = runtime[:, n_i].mean()
            std = runtime[:, n_i].std(ddof=1)
            lower_ci = mean - Z2STDEV * std / (repeats**0.5)
            if timeout < lower_ci:
                runtime[:, (n_i + 1) :] = None
                if prog_bar:
                    prog_bar.update(max(len(ns) - n_i - 1, 0) * repeats)
                break

    mean_runtime = runtime.mean(axis=0)
    std_runtime = runtime.std(axis=0, ddof=1)

    return (mean_runtime, std_runtime)


def compare_runtime_sparse_vs_dense(
    length: float,
    ns: npt.NDArray,
    repeats: int,
    z_score: float = 1.97,
    timeout: float | None = 1.0,
    ax: Axes | None = None,
) -> Axes:
    """Produce a plot comparing the runtime performance of different eigensolvers.

    Args:
        length: Physical side length of the object.
        ns: Numpy array of values to test for N, the number of spatial intervals
            used to divide the cartesian axes.
        repeats: Number of repeats used to calculate the sample standard deviation.
        z_score: Z-score used to calculate the confidence intervals around mean runtime.
        timeout: Maximum runtime before termination.
        ax: Optional Matplotlib axis on which to create plot. If not provided, function
            uses the current artist.

    Returns:
        Matplotlib artist containing the comparison plot.
    """
    if ax is None:
        ax = plt.gca()

    with tqdm(total=(len(ns) * repeats * 4)) as prog_bar:
        mean_dense, std_dense = measure_runtime(
            length,
            ns,
            repeats,
            use_sparse=False,
            timeout=timeout,
            prog_bar=prog_bar,
        )
        mean_sparse_eigs, std_sparse_eigs = measure_runtime(
            length,
            ns,
            repeats,
            use_sparse=True,
            shift_invert=False,
            use_eigsh=False,
            timeout=timeout,
            prog_bar=prog_bar,
        )
        mean_sparse, std_sparse = measure_runtime(
            length,
            ns,
            repeats,
            use_sparse=True,
            shift_invert=False,
            use_eigsh=True,
            timeout=timeout,
            prog_bar=prog_bar,
        )
        mean_sparse_shift_inv, std_sparse_shift_inv = measure_runtime(
            length,
            ns,
            repeats,
            use_sparse=True,
            shift_invert=True,
            use_eigsh=True,
            timeout=timeout,
            prog_bar=prog_bar,
        )

    ci_dense = z_score * std_dense / (repeats**0.5)
    ci_sparse = z_score * std_sparse / (repeats**0.5)
    ci_sparse_eigs = z_score * std_sparse_eigs / (repeats**0.5)
    ci_sparse_shift_inv = z_score * std_sparse_shift_inv / (repeats**0.5)

    ax.plot(ns, mean_dense, label="la.eigh")
    ax.fill_between(
        ns,
        mean_dense - ci_dense,
        mean_dense + ci_dense,
        alpha=0.7,
    )

    ax.plot(ns, mean_sparse_eigs, label="sp.eigs")
    ax.fill_between(
        ns,
        mean_sparse_eigs - ci_sparse_eigs,
        mean_sparse_eigs + ci_sparse_eigs,
        alpha=0.7,
    )

    ax.plot(ns, mean_sparse, label="sp.eigsh")
    ax.fill_between(
        ns,
        mean_sparse - ci_sparse,
        mean_sparse + ci_sparse,
        alpha=0.7,
    )

    ax.plot(ns, mean_sparse_shift_inv, label="sp.eigsh (SI)")
    ax.fill_between(
        ns,
        mean_sparse_shift_inv - ci_sparse_shift_inv,
        mean_sparse_shift_inv + ci_sparse_shift_inv,
        alpha=0.7,
    )

    ax.set_xlabel("Spatial discretisation intervals (N)")
    ax.set_ylabel("Runtime (s)")
    ax.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8,
        handlelength=1,
        columnspacing=0.5,
        labelspacing=0.2,
        frameon=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, None)

    return ax

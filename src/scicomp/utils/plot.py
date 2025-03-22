"""Plotting functions."""

import itertools
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.patches import Patch

from scicomp.domains import Domain, ShapeEnum


def configure_mpl():
    """Configure Matplotlib style."""
    FONT_SIZE_SMALL = 8
    FONT_SIZE_DEFAULT = 10
    FONT_PATH = Path("fonts/LibertinusSerif-Regular.otf")
    font_manager.fontManager.addfont(FONT_PATH)

    plt.rc("font", family="Libertinus Serif")
    plt.rc("font", weight="normal")  # controls default font
    plt.rc("mathtext", fontset="stix")
    plt.rc("font", size=FONT_SIZE_DEFAULT)  # controls default text sizes
    plt.rc("axes", titlesize=FONT_SIZE_DEFAULT)  # fontsize of the axes title
    plt.rc("axes", labelsize=FONT_SIZE_DEFAULT)  # fontsize of the x and y labels
    plt.rc("figure", labelsize=FONT_SIZE_DEFAULT)
    plt.rc("figure", dpi=600)

    sns.set_context(
        "paper",
        rc={
            "axes.linewidth": 0.5,
            "axes.labelsize": FONT_SIZE_DEFAULT,
            "axes.titlesize": FONT_SIZE_DEFAULT,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "ytick.minor.width": 0.4,
            "xtick.labelsize": FONT_SIZE_SMALL,
            "ytick.labelsize": FONT_SIZE_SMALL,
        },
    )


def plot_eigenmode(
    mode: npt.NDArray[np.float64],
    freq: float,
    width: float,
    height: float,
    index_grid: npt.NDArray[np.float64],
    origin: tuple[float, float] = (0.0, 0.0),
    ax: Axes | None = None,
) -> AxesImage:
    """Plot an eigenmode of a circular drum as a 2D heatmap.

    Args:
        mode: Eigenmode as a vector.
        freq: Eigenfrequency (sqrt(-K)) associated with the eigenmode.
        width: Width of the shape.
        height: Height of the shape.
        nx: Number of discretisation intervals used to divide the x-axis.
        ny: Number of discretisation intervals used to divide the y-axis.
        index_grid: Matrix with shape NxN, with NaN in cells outside the circular
            drum, and contiguous cell indexes in cells within the drum.
        origin: Lower left corner of the shape in physical coordinates.
        ax: (Optional) Matplotlib axis to plot onto. If not supplied, plot will
            use the current global artist.

    Returns:
        AxesImage with 2D heatmap of the eigenmode.
    """
    if ax is None:
        ax = plt.gca()

    ny, nx = index_grid.shape

    grid = np.full((ny, nx), np.nan)
    grid[~np.isnan(index_grid)] = mode
    y0, x0 = origin
    im_ax = ax.imshow(
        grid,
        extent=(x0, width, y0, height),
        origin="lower",
        cmap="bwr",
    )
    ax.set_title(f"$\\omega = {freq:.2f}$", fontsize=9)

    return im_ax


def plot_shape_eigenmodes(
    domain: Domain,
    n: int,
    use_sparse: bool,
    shift_invert: bool,
    axes,
):
    """Plot first `k` eigenmodes for a 2D Domain as heatmaps.

    Args:
        domain: 2D domain (shape).
        n: Spatial discretisation resolution.
        use_sparse: Use sparse eigenvalue solver.
        shift_invert: Use shift-invert method to improve performance when
            identifying small magnitude eigenvalues. Only applicable for
            sparse solvers.
        axes: Row of matplotlib axes to plot onto. Length of row must == `k`.
    """
    k = len(axes)
    index_grid = domain.discretise(n)
    eigenfrequencies, eigenmodes = domain.solve_eigenproblem(
        k=k,
        ny=n,
        use_sparse=use_sparse,
        shift_invert=shift_invert,
        index_grid=index_grid,
    )
    for i, ax in enumerate(axes):
        plot_eigenmode(
            eigenmodes[:, i],
            eigenfrequencies[i],
            float(domain.width),
            float(domain.height),
            index_grid,
            ax=ax,
        )
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


def plot_eigenspectrum_by_length(
    n_at_unit_length: int,
    palette: list[tuple[float, float, float]] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Show spectrum of eigenfrequencies as function of shape length.

    Plot is displayed as a violinplot, truncated at the range of the data.
    For each length the distribution of eigenfrequencies for each shape is
    shows.

    Maintans a consistent spatial stepsize h, by varying the number of
    spatial intervals used to discretise the domains linearly with the
    shape length.

    Args:
        n_at_unit_length: Number of spatial discretisation intervals at L=1
        palette: Optional palette to use for the different shapes.
        ax: Optional Matplotlib axis. If none is provided, uses current artist.

    Returns:
        Matplotlib axis containing violinplot.
    """
    if ax is None:
        ax = plt.gca()

    if palette is None:
        palette = sns.color_palette()[: len(ShapeEnum)]

    h = Fraction(1, n_at_unit_length)
    k = n_at_unit_length - 1

    lengths = [Fraction(length, 10) for length in np.arange(10, 51, 20)]

    data = {"length": [], "omega": [], "shape": []}
    for length, shape in itertools.product(lengths, reversed(ShapeEnum)):
        width = length * 2 if shape == ShapeEnum.Rectangle else length
        domain = shape.domain(width=width, height=length)
        n = (length / h).numerator
        eigenfrequencies, _ = domain.solve_eigenproblem(
            k, n, use_sparse=True, shift_invert=True
        )
        data["length"].extend(np.repeat(float(length), len(eigenfrequencies)).tolist())
        data["omega"].extend(eigenfrequencies.tolist())
        data["shape"].extend([shape] * len(eigenfrequencies))

    sns.violinplot(
        x=data["length"],
        y=data["omega"],
        hue=data["shape"],
        orient="v",
        width=0.75,
        linewidth=0.5,
        cut=0,
        bw_adjust=0.4,
        common_norm=False,
        density_norm="count",
        palette=palette,
        saturation=0.5,
        ax=ax,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Object length ($m$)")
    ax.set_ylabel("$\\omega$ ($\\text{rad}/s$)")
    handles = [Patch(facecolor=c, linewidth=0.5, edgecolor="black") for c in palette]
    ax.legend(
        handles=handles,
        labels=["Circle", "Square", "Rect."],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8,
        handlelength=0.5,
        labelspacing=1,
        columnspacing=0.75,
        frameon=False,
    )
    return ax


def plot_eigenspectrum_by_n(
    min_n: int,
    max_n: int,
    palette: list[tuple[float, float, float]] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Show spectrum of eigenfrequencies as function of discretisation resolution.

    Plot is displayed as a violinplot, truncated at the range of the data.
    For each shape the distribution of eigenfrequencies is shown for two different
    spatial discretisations (`min_n` and `max_n`).

    The same shape length is used in both cases (L = 1).

    Args:
        min_n: Lower spatial discretisation resolution.
        max_n: Upper spatial discretisation resolution.
        palette: Optional palette to use for the different shapes.
        ax: Optional Matplotlib axis. If none is provided, uses current artist.

    Returns:
        Matplotlib axis containing violinplot.
    """
    if ax is None:
        ax = plt.gca()

    if palette is None:
        palette = sns.color_palette()[:2]

    length = 1

    k1 = min_n - 1

    data = {
        "n": [],
        "lambda": [],
        "first_k1": [],
        "shape": [],
    }
    ns = [min_n, max_n]
    for n, shape in itertools.product(ns, reversed(ShapeEnum)):
        width = length * 2 if shape == ShapeEnum.Rectangle else length
        domain = shape.domain(width=width, height=length)

        k = n - 1
        eigenfrequencies, _ = domain.solve_eigenproblem(
            k, n, use_sparse=True, shift_invert=True
        )
        data["n"].extend([str(n)] * len(eigenfrequencies))
        data["lambda"].extend(eigenfrequencies.tolist())
        first_k_mask = np.arange(len(eigenfrequencies)) >= k1
        data["first_k1"].extend(first_k_mask.tolist())
        data["shape"].extend([shape.title()] * len(eigenfrequencies))

    sns.violinplot(
        x=data["shape"],
        y=data["lambda"],
        hue=data["n"],
        orient="v",
        width=0.75,
        linewidth=0.5,
        cut=0,
        bw_adjust=0.4,
        common_norm=False,
        density_norm="count",
        saturation=0.5,
        split=True,
        inner="stick",
        inner_kws={"linewidths": 0.15, "colors": "0.8"},
        palette=palette,
        ax=ax,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel("$\\omega_i$ ($\\text{rad}/s$)")

    handles = [Patch(facecolor=c, linewidth=0.5, edgecolor="black") for c in palette]
    ax.legend(
        handles=handles,
        labels=[f"$N={n}$" for n in ns],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=0.5,
        fontsize=8,
        labelspacing=0.1,
        columnspacing=0.75,
        frameon=False,
    )

    return ax

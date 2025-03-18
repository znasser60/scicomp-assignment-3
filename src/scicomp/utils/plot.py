"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def configure_mpl():
    """Configure Matplotlib style."""
    FONT_SIZE_SMALL = 8
    FONT_SIZE_DEFAULT = 10

    plt.rc("font", family="Georgia")
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
    length: float,
    n: int,
    index_grid: npt.NDArray[np.int64],
    ax: Axes | None = None,
) -> AxesImage:
    """Plot an eigenmode of a circular drum as a 2D heatmap.

    Args:
        mode: Eigenmode as a vector.
        freq: Eigenfrequency (sqrt(-K)) associated with the eigenmode.
        length: Diameter of the circular drum.
        n: Number of discretisation intervals used to divide the cartesian axes.
        index_grid: Matrix with shape NxN, with NaN in cells outside the circular
            drum, and contiguous cell indexes in cells within the drum.
        ax: (Optional) Matplotlib axis to plot onto. If not supplied, plot will
            use the current global artist.

    Returns:
        AxesImage with 2D heatmap of the eigenmode.
    """
    if ax is None:
        ax = plt.gca()

    grid = np.full((n, n), np.nan)
    grid[~np.isnan(index_grid)] = mode
    radius = length / 2
    im_ax = ax.imshow(
        grid,
        extent=(-radius, radius, -radius, radius),
        origin="lower",
        cmap="bwr",
    )
    extra_space = 1.1
    ax.set_xlim(-radius * extra_space, radius * extra_space)
    ax.set_ylim(-radius * extra_space, radius * extra_space)
    ax.set_title(f"λ={freq:.4f}")

    return im_ax

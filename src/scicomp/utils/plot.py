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
    # extra_space = 1.1
    # ax.set_xlim(-radius * extra_space, radius * extra_space)
    # ax.set_ylim(-radius * extra_space, radius * extra_space)
    ax.set_title(f"λ={freq:.4f}")

    return im_ax

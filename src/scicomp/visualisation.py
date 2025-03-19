"""Plotting functions to analyse simulations."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from scicomp.shape import Shape


def plot_rectangular_state(
    z: npt.NDArray[np.float64], width: float, height: float, ax: Axes | None = None
) -> AxesImage:
    """Plot the values of a rectangular grid as a heatmap.

    Values may be amplitude, concentration, or any other 1D quantity associated
    with a grid.

    Args:
        z: Value at each grid node.
        width: Width of the physical grid in SI units.
        height: Height of the physical grid in SI units.
        ax: Matplotlib axis. If not provided, defaults to the current global axis.

    Returns:
        Matplotlib AxesImage with heatmap of grid values.
    """
    if ax is None:
        ax = plt.gca()
    ax_im = ax.imshow(z, extent=(0, width, 0, height), origin="lower")
    return ax_im


def plot_circular_state(
    z: npt.NDArray[np.float64],
    diameter: float,
    n: int,
    center: tuple[float, float] = (0.0, 0.0),
    ax: Axes | None = None,
) -> AxesImage:
    """Plot the values of a circular grid as a heatmap.

    Values may be amplitude, concentration, or any other 1D quantity associated
    with a grid.

    Args:
        z: Value at each grid node.
        diameter: Diameter of the physical grid in SI units.
        n: Number of grid points used to discretise the diameter.
        center: Tuple containing (xc, yc), the coordinates of the circle center.
        ax: Matplotlib axis. If not provided, defaults to the current global axis.

    Returns:
        Matplotlib AxesImage with heatmap of grid values.
    """
    if ax is None:
        ax = plt.gca()

    # Set points outside circle to None
    radius = diameter / 2
    X, Y = np.meshgrid(
        np.linspace(-radius, radius, n + 1), np.linspace(-radius, radius, n + 1)
    )
    mask = radius**2 > X**2 + Y**2
    z = z.copy()
    z[~mask] = None

    xc, yc = center
    ax_im = ax.imshow(
        z, extent=(xc - radius, diameter, yc - radius, diameter), origin="lower"
    )
    return ax_im


def plot_eigenmode(
    eigenmode: npt.NDArray,
    eigenfrequency: float,
    shape: Shape,
    ax: Axes | None = None,
    **kwargs,
) -> AxesImage:
    """Plot an eigenmode of a given shape.

    Args:
        eigenmode: Numpy vector representing the eigenmode values in a state vector.
        eigenfrequency: Eigenvalue associated with the eigenmode.
        shape: Shape of the associated object.
        ax: Matplotlib axis. If not provided, defaults to the current global axis.
        **kwargs: Additional arguments to be passed to `plot_circular_state`
            or `plot_rectangular_state`.

    Returns:
        Matplotlib AxesImage with heatmap of eigenmode.
    """
    if ax is None:
        ax = plt.gca()
    grid_values = shape.state_vec_to_grid(eigenmode)
    match shape:
        case Shape.Circle:
            im_ax = plot_circular_state(grid_values, ax=ax, **kwargs)
        case _:
            im_ax = plot_rectangular_state(grid_values, ax=ax, **kwargs)
    ax.set_title(f"$\\lambda = {eigenfrequency}$")
    return im_ax

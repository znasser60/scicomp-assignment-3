"""Functions for finite difference verification."""

import math
from fractions import Fraction

import numpy as np
import numpy.typing as npt


def check_valid_spatial_discretisation(
    width: Fraction | int, height: Fraction | int, n: int
) -> bool:
    """Verify that N gives a valid discretisation of a given rectangle.

    Assuming that dy=dx, discretisation of a rectangle using N grid points
    should yield an integer number of grid points in the y-direction, with
    the boundaries explicitly represented.

    We verify this by calculating dx = width / N, and checking that
    dx does indeed divide height exactly. Fractions are used to prevent
    floating-point arithmetic errors.

    Args:
        width: Physical width of the rectangle.
        height: Physical height of the rectangle.
        n: Number of grid points with which to discretise the x-axis.

    Returns:
        Boolean value indicating that N yields (or does not yield) a valid
        scheme.
    """
    width = Fraction(width)
    height = Fraction(height)
    dx = Fraction(numerator=width, denominator=n)
    dy = height * (1 / dx)
    return dy.denominator == 1


def check_z_folder_input(mask: npt.NDArray[np.bool]) -> None:
    """Validates input compatibility with the z-order indexing process.

    Args:
        mask: 2D NumPy array containing boolean values indicating whether each cell
                belongs to the original shape or not.

    Returns:
        None
    """
    check_if_mask_rectangular(mask)

    # Removing any row or columns that is not part of the original rectangle shape
    domain_shape = mask[np.any(mask, axis=1), :][:, np.any(mask, axis=0)]

    m, n = domain_shape.shape
    check_if_grid_bigger_than_two(m, n)
    min_val, max_val = np.min([m, n]), np.max([m, n])
    check_if_right_4_mult(min_val)
    if max_val % min_val != 0:
        raise ValueError(f"Grid size '{max_val}' must be a multiple of '{min_val}'.")


def check_if_mask_rectangular(mask: npt.NDArray[np.bool]) -> None:
    """Raises error if the mask is not representing a rectangle.

    Note: A small resolute circle whose mask appears to be a rectangle is not filtered.

    Args:
        mask: 2D NumPy array containing boolean values indicating whether each cell
                belongs to the original shape or not.

    Returns:
        None
    """
    n_cell_per_row = np.sum(mask, axis=1)
    if not np.isin(n_cell_per_row, [0, np.max(n_cell_per_row)]).all():
        raise ValueError("The mask shape must be a rectangle.")

    n_cell_per_col = np.sum(mask, axis=0)
    if not np.isin(n_cell_per_col, [0, np.max(n_cell_per_col)]).all():
        raise ValueError("The mask shape must be a rectangle.")


def check_if_right_4_mult(size_to_check: int) -> None:
    """Checks if the grid size is in the required form.

    The form is 4 * 2^n, where n is a non-negative integer.

    Args:
        size_to_check: Input value on which the test conditions are checked

    Returns:
        None
    """
    if size_to_check % 4 != 0:
        raise ValueError(
            f"'Grid size: {size_to_check}' must be in a form 4 * 2^n, where n is a "
            f"non-negative integer."
        )
    size_remainder = size_to_check / 4
    if not math.log2(size_remainder).is_integer():
        raise ValueError(
            f"Grid size: '{size_to_check}' must be in a form 4 * 2^n, where n is a "
            f"non-negative integer."
        )


def check_if_grid_bigger_than_two(m: int, n: int) -> None:
    """Checks if the shapes of the gird are at least 2.

    Args:
        m: Height of the rectangular grid
        n: Width of the rectangular grid

    Returns:
        None
    """
    if m < 2:
        raise ValueError("Grid height 'm' must be at least 2.")
    if n < 2:
        raise ValueError("Grid width 'n' must be at least 2.")

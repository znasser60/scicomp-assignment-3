"""Functions for finite difference verification."""

from fractions import Fraction


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

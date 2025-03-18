"""Create indexing for variously shaped surfaces."""

import logging
import numpy as np

from scicomp.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def initialize_circular_grid(length: float, n: int):
    """Initializes a 2D grid representing a circular domain.

    This function creates a square N x N grid representing a domain with a circular
    region of diameter L. The points inside the circle are assigned unique index values
    starting from 0, while points outside the circle are set to None.

    Args:
        length: Length of the original surface.
        n: Resolution of the discretization.

    Returns:
    mask : ndarray
        Boolean mask indicating which points are inside the circular domain.

    index_grid : ndarray
        2D array with indices starting from 0 for points inside the circle, and None
        for points outside.
    """
    x_range = np.linspace(-length / 2, length / 2, n)
    y_range = np.linspace(-length / 2, length / 2, n)
    x, y = np.meshgrid(x_range, y_range)
    mask = x ** 2 + y ** 2 < (length / 2) ** 2
    index_grid = np.full((n, n), np.nan, dtype=np.float64)
    index_grid[mask] = np.arange(np.sum(mask))

    logger.info("Indexing for circular domain has been created successfully.")

    return index_grid

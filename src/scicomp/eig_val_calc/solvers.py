"""Space equation solvers for various shapes."""

import logging

from scicomp.eig_val_calc.equation_solver_components.create_indexing import (
    initialize_circular_grid,
)
from scicomp.eig_val_calc.equation_solver_components.solving_equation import (
    solve_laplacian,
)
from scicomp.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def solve_circle_laplacian(
    length,
    n: int,
    k: int | None = None,
    use_sparse: bool = False,
    shift_invert: bool = False,
):
    """Solve eigenvalue Laplacian on a circular domain.

    The sparse Laplacian matrix is built such that the main diagonal is set to -4,
    and the adjacent neighbors (up, down, left, right) are set to 1.

    Returns:
        Tuple containing the eigenfrequencies, eigenmodes (eigenvectors), and a
        2D array with indices of cells which lie within the circle. The index matrix
        is None for points outside the circle.
    """
    circular_index_grid = initialize_circular_grid(length, n)

    frequencies, eigenmodes = solve_laplacian(
        length, n, circular_index_grid, k, use_sparse, shift_invert
    )

    logger.info("Laplacian space equation is solved for circular domain.")

    return frequencies, eigenmodes, circular_index_grid

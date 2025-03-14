"""Functions that create the index scheme and calculate the lintrans of the Laplace op.

The index schemes are calculated for various shaped rectangle and circular grids.
The linear transformation equivalent of the Laplace operator is calculated for the
discretised function v(x, y), to solve the following equation: Δv = Mv = Kv.
"""

import math

import numba as nb
import numpy as np
import numpy.typing as npt


def create_rectangle_row_wise_indexing(m: int, n: int) -> npt.NDArray[np.int32]:
    """Creates an m x n rectangular grid indexing scheme using row-wise indexing.

    Uses a row-wise indexing method to create an array of the same dimensions as the
    original grid, where each cell is assigned its corresponding row-wise index.

    For example:
    -4 by 4:
                1  2  3  4
                5  6  7  8
                9  10 11 12
                13 14 15 16

    Args:
        m: Height of the rectangular grid
        n: Width of the rectangular grid

    Returns:
        2D numpy grid containing the corresponding row-wise index
    """
    return np.reshape(np.arange(m * n), (m, n))


def create_rectangle_z_ordered_indexing(m: int, n: int) -> npt.NDArray[np.int32]:
    """Creates an indexing for a rectangular grid using the z-order technique.

    Uses a row-wise indexing method to create an array of the same dimensions as the
    original grid, where each cell is assigned its corresponding row-wise index.

    For example:
    - 2 by 2:
                1 2
                3 4
    - 4 by 4:
                1  2  5  6
                3  4  7  8
                9  10 13 14
                11 12 15 16
    - 2 by 4:
                1 2 5 6
                3 4 7 8
    - 4 by 2:
                1 2
                3 4
                5 6
                7 8

    Args:
        m: Height of the rectangular grid
        n: Width of the rectangular grid

    Returns:
        2D numpy grid containing the corresponding z-order indexing.
    """
    check_z_folder_input(m, n)

    # Create sub-square
    current_grid = np.array([[0, 1], [2, 3]])
    for _ in range(int(np.min([m, n]) / 2 - 1)):
        shift_scaler = current_grid[-1, -1] + 1
        current_grid = np.block(
            [
                [current_grid, current_grid + shift_scaler],
                [current_grid + shift_scaler * 2, current_grid + shift_scaler * 3],
            ]
        )

    # Add the rest of the rectangle
    if m < n:
        repeat_counter = int(n / m)
        for _ in range(repeat_counter - 1):
            shift_scaler = current_grid[-1, -1]
            current_grid = np.block([[current_grid, current_grid + shift_scaler]])

    return current_grid


@nb.njit
def calc_rectangle_grid_mat(indexing: npt.NDArray[np.int32]) -> npt.NDArray[np.int8]:
    # Potential np.float64 for future compatibility
    """Calculates the linear transformation equivalent of the Laplace operator.

     Given a grid and an indexing scheme, the function computes the linear
     transformation equivalent of the Laplace operator on a discretized v(x, y)
     function, to solve the equation: Δv = Mv = Kv.

    Args:
        indexing: 2D numpy array, which has the same shape as the original grid, and
                    contains the cell indices in the corresponding coordinates

    Returns:
        M matrix in the Mv = Kv equation. (2D numpy array)
    """
    m_mat = np.eye(indexing.shape[0] * indexing.shape[1], dtype=np.int8) * -4

    for i in range(m_mat.shape[0]):
        # Get the coordinate (real numpy coordinate) of the current
        row_coord, col_coord = np.where(indexing == i)
        row_coord, col_coord = row_coord[0], col_coord[0]

        neumann_neigh_indices = np.array(
            [
                indexing[max(0, row_coord - 1), col_coord],  # top neighbour
                indexing[
                    min(indexing.shape[0] - 1, row_coord + 1), col_coord
                ],  # bottom neighbour
                indexing[row_coord, max(0, col_coord - 1)],  # left neighbour
                indexing[
                    row_coord, min(indexing.shape[1] - 1, col_coord + 1)
                ],  # right neighbour
            ]
        )
        neumann_neigh_indices = neumann_neigh_indices[neumann_neigh_indices != i]

        m_mat[i, neumann_neigh_indices] = 1

    return m_mat


def check_z_folder_input(m: int, n: int) -> None:
    """Validates input compatibility with the z-order indexing process.

    Args:
        m: Height of the rectangular grid
        n: Width of the rectangular grid

    Returns:
        None
    """
    check_if_grid_bigger_than_two(m, n)
    min_val, max_val = np.min([m, n]), np.max([m, n])
    check_if_right_4_mult(min_val)
    if max_val % min_val != 0:
        raise ValueError(f"Grid size '{max_val}' must be a multiple of '{min_val}'.")


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

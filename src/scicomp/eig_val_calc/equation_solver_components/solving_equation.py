"""Frequency, eigenmod calculators and u(x, y, t) = v(x, y)T(t) evaluator."""

import logging
from typing import Callable, Tuple, Union, Optional
from functools import partial

import numpy as np
import numpy.typing as npt
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

from scicomp.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def solve_laplacian(
    length: float,
    n: int,
    index_grid: npt.NDArray[np.int64],
    k: Optional[int] = None,
    use_sparse: bool = False,
    shift_invert: bool = False,
):
    """Solve eigenvalue Laplacian on a circular domain.

    The sparse Laplacian matrix is built such that the main diagonal is set to -4,
    and the adjacent neighbors (up, down, left, right) are set to 1.

    Args:
        length: Length of the real surface side.
        n: Resolution of the discretisation.
        index_grid: Discretised grid of the original surface, where each cell contains
                        np.nan if not part of the original shape, otherwise the
                        indexing value, used to create eigenmod vector.
        k: Number of eigenvalues to compute.
        use_sparse: Whether to use sparse Laplacian or not.
        shift_invert: Whether to shift the eigenvalues of the Laplacian

    Returns:
        Tuple containing the eigenfrequencies, eigenmodes (eigenvectors), and a
        2D array with indices of cells which lie within the circle. The index matrix
        is None for points outside the circle.
    """
    if k is None:
        k = n - 1

    # Calculate the index grid for the chosen shape
    num_circle_points = np.sum(~np.isnan(index_grid))
    eig_solver, laplacian = set_up_eig_solver_solver(
        num_circle_points,
        k,
        use_sparse,
        shift_invert,
    )

    # Calculate Laplacian matrix
    compute_laplacian_matrix(index_grid, laplacian)

    # Solve the equation
    frequencies, eigenmodes = solve_eig_equation(
        laplacian,
        eig_solver,
        k,
        length / n
    )

    return frequencies, eigenmodes


def set_up_eig_solver_solver(
        matrix_size_length: int,
        k: int,
        use_sparse: bool = False,
        shift_invert: bool = False
) -> Tuple[Callable, npt.NDArray[np.float64]]:
    """Sets up the eigenvalue solver function and the base for the matrix.

    Args:
        matrix_size_length: Size length of the square matrix.
        use_sparse: Boolean to use sparse solver of not.
        shift_invert: Boolean to use shift inverse or not.
        **kwargs: Additional keywords argument for the solver.

    Returns:
        Solver function and the base matrix.
    """
    if use_sparse:
        if shift_invert:
            eig_solver = partial(sp_la.eigsh, k=k, sigma=0, v0=np.ones(matrix_size_length))
        else:
            eig_solver = partial(sp_la.eigsh, k=k, which="SM", v0=np.ones(matrix_size_length))
        laplacian = sp.lil_matrix(
            (matrix_size_length, matrix_size_length), dtype=np.float64
        )
    else:
        eig_solver = la.eigh
        laplacian = np.zeros((matrix_size_length, matrix_size_length), dtype=np.float64)

    return eig_solver, laplacian


def compute_laplacian_matrix(
        index_grid: npt.NDArray[np.int64],
        laplacian: Union[npt.NDArray[np.float64], sp.lil_matrix],
) -> None:
    """Computes the Laplacian matrix (in place).

    Args:
        index_grid: Discretised grid of the original surface, where each cell contains
                        np.nan if not part of the original shape, otherwise the
                        indexing value, used to create eigenmod vector.
        laplacian: Prepared base for the matrix that replace Lalplacian operator in the
                        discretised case.

    Returns:
        Final Laplacian matrix.
    """
    # Calculate the possible value space sides
    n, m = index_grid.shape

    # Compute the Laplacian matrix
    for i in range(n):
        for j in range(n):
            # Check if the cell is part of the original shape
            if not np.isnan(index_grid[i, j]):
                idx = int(index_grid[i, j])

                # Fill the main diagonal
                laplacian[idx, idx] = -4

                # Find the fill the neighbour values in the row of the corresp. cell
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj

                    # Check if the neighbour exists (if not bound condition applied)
                    if 0 <= ni < n and 0 <= nj < n and not np.isnan(index_grid[ni, nj]):
                        laplacian[idx, int(index_grid[ni, nj])] = 1


def solve_eig_equation(
        laplacian: Union[npt.NDArray[np.float64], sp.lil_matrix],
        eig_solver: Callable,
        k: int,
        h: float
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculates the frequencies and the eigenmods.

    The equation the function solves is assumed to have the form: Mv = Kv, where M is
    the laplacian matrix and -K is the λ^2.

    Args:
        laplacian: Matrix that performs the laplace operator on the discretised grid.
        eig_solver: Chosen function to solce the equation
        k: Number of eigenvalues to compute.
        h: Stepsize of the discretisation.

    Returns:
        Frequencies (λ) (1D numpy array) and eigenmods (v) (2D numpy array).
    """
    # Scale the laplacian matrix
    scaled_laplacian = laplacian / h**2

    # Calculate frequencies and eigenmods
    eigenvalues, eigenvectors = eig_solver(scaled_laplacian)
    eigenfrequencies = (-eigenvalues) ** 0.5

    # Determine how many of the should be returned
    sort_idx = np.argsort(np.abs(eigenvalues))[:k]

    return eigenfrequencies[sort_idx], eigenvectors[:, sort_idx]


def eval_oscillating_solution(
    t: float | int,
    mode: npt.NDArray[np.float64],
    freq: float,
    c: float = 1.0,
    cosine_coef: float | int = 1.0,
    sine_coef: float | int = 1.0,
) -> npt.NDArray[np.float64]:
    """Evaluate an eigensolution of the form K < 0 at a particular time.

    Args:
        t: Time-point to evaluate.
        mode: Eigenmode as a vector.
        freq: Eigenfrequency (sqrt(-K)) associated with the eigenmode.
        c: Wave propagation velocity (m/s).
        cosine_coef: Multiplication coefficient for the cosine term in T(t).
        sine_coef: Multiplication coefficient for the sine term in T(t).

    Returns:
        Amplitude at time t, given the input eigenmode and eigenfrequency.
        The returned vector has the same shape as the mode parameter.
    """
    if freq <= 0:
        raise ValueError(f"Expected eigenfrequency λ > 0, received {freq:.4f}")

    cosine_component = cosine_coef * np.cos(c * freq * t)
    sine_component = sine_coef * np.sin(c * freq * t)
    return mode * (cosine_component + sine_component)

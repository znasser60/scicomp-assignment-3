"""Solves Eigenmodes and steady-state diffusion on a circular domain."""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def initialize_grid(length, n):
    """Initializes a 2D grid representing a circular domain.

    This function creates a square N x N grid representing a domain with a circular
    region of diameter L. The points inside the circle are assigned unique index values
    starting from 0, while points outside the circle are set to None.

    Returns:
    mask : ndarray
        Boolean mask indicating which points are inside the circular domain.

    index_grid : ndarray
        2D array with indices starting from 0 for points inside the circle, and None
        for points outside.
    """
    x = np.linspace(-length / 2, length / 2, n)
    y = np.linspace(-length / 2, length / 2, n)
    X, Y = np.meshgrid(x, y)
    mask = (length / 2) ** 2 > X**2 + Y**2
    index_grid = np.full((n, n), np.nan, dtype=float)
    circle_points = np.where(mask)
    num_circle_points = len(circle_points[0])
    index_grid[circle_points] = np.arange(num_circle_points)

    return mask, index_grid


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

    length: Diameter of the circle domain
    n: The number of grid points in each dimension
    k: The number of eigenvalues/eigenvectors to compute for 
        the Laplacian matrix. If None, it defaults to n-1.
    use_sparse: If True, the sparse matrix solver is used 
                for efficiency.
    shift_invert: If True, a shift-invert method is used 
                    in sparse eigenvalue computations.
        
    Returns:
        Tuple containing the eigenfrequencies, eigenmodes (eigenvectors), a
        2D array with indices of cells which lie within the circle, 
        and the Laplacian matrix M. The index matrix is None for points outside 
        the circle.
    """
    if k is None:
        k = n - 1
    mask, index_grid = initialize_grid(length, n)
    num_circle_points = np.count_nonzero(mask)
    if use_sparse:
        if shift_invert:
            eig_solver = partial(sp_la.eigsh, k=k, sigma=0)
        else:
            eig_solver = partial(sp_la.eigsh, k=k, which="SM")
        laplacian = sp.lil_matrix(
            (num_circle_points, num_circle_points), dtype=np.float64
        )
    else:
        eig_solver = la.eigh
        laplacian = np.zeros((num_circle_points, num_circle_points), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if not np.isnan(index_grid[i, j]):
                idx = int(index_grid[i, j])
                laplacian[idx, idx] = -4
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and not np.isnan(index_grid[ni, nj]):
                        laplacian[idx, int(index_grid[ni, nj])] = 1

    h = length / n
    laplacian /= h**2
    eigenvalues, eigenvectors = eig_solver(laplacian)
    eigenfrequencies = (-eigenvalues) ** 0.5

    sort_idx = np.argsort(np.abs(eigenvalues))[:k]

    return eigenfrequencies[sort_idx], eigenvectors[:, sort_idx], index_grid, laplacian


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

def solve_circle_diffusion(
    source_position: tuple, 
    length,
    n: int,
    k: int | None = None, 
    use_sparse: bool = False,
    shift_invert: bool = False,
    ): 
    """Solves the steady state diffusion equation using direct methods (Mc = b).

    Args:
        source_position: Specified grid position of the source 
        length: Diameter of the circle domain
        n: The number of grid points in each dimension
        k: The number of eigenvalues/eigenvectors to compute for 
            the Laplacian matrix. If None, it defaults to n-1.
        use_sparse: If True, the sparse matrix solver is used 
                    for efficiency.
        shift_invert: If True, a shift-invert method is used 
                        in sparse eigenvalue computations.
    
    Returns:
        c_grid: A 2D array representing the concentration distribution across the grid
                on a circle domain.

    """
    mask, index_grid = initialize_grid(length,n)
    _, _, _, laplacian = solve_circle_laplacian(length, n, k, use_sparse, shift_invert)
    num_circle_points = np.count_nonzero(mask)
    b = np.zeros(num_circle_points)
    x, y = np.linspace(-length/2, length/2, n), np.linspace(-length/2, length/2, n)
    source_idx = index_grid[np.argmin(np.abs(y - source_position[1])), 
                            np.argmin(np.abs(x - source_position[0]))]
    
    source_idx = int(source_idx)
    laplacian[source_idx, :] = 0  
    laplacian[source_idx, source_idx] = 1  
    b[source_idx] = 1

    c = sp_la.spsolve(laplacian, b) if use_sparse else la.solve(laplacian, b)

    c_grid = np.full((n, n), np.nan)
    c_grid[mask] = c 

    return c_grid

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

def plot_circle_diffusion(n, c_grid, length):
    """Plots the steady-state concentration solution on the circular domain."""
    x = np.linspace(-length / 2, length / 2, n)
    y = np.linspace(-length / 2, length / 2, n)
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, c_grid, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(label="Concentration c(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
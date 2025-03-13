import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

def initialize_grid(L, N): 
    """
    Initializes a 2D grid representing a circular domain.

    This function creates a square N x N grid representing a domain with a circular region 
    of diameter L. The points inside the circle are assigned unique index values starting from 0, 
    while points outside the circle are set to None.

    Returns: 
    mask : ndarray
        Boolean mask indicating which points are inside the circular domain. 

    index_grid : ndarray 
        2D array with indices starting from 0 for points inside the circle, and None 
        for points outside.
    """
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    mask = X**2 + Y**2 < (L/2)**2 
    index_grid = np.full((N, N), None)
    circle_points = np.where(mask)
    num_circle_points = len(circle_points[0])
    index_grid[circle_points] = np.arange(num_circle_points)

    return mask, index_grid 

def solve_circle_laplacian(L, N): 
    """
    Constructs and solves the Laplace operator on a discretized circle 
    using the central difference method.

    The sparse Laplacian matrix is built such that the main diagonal is set to -4, 
    and the adjacent neighbors (up, down, left, right) are set to 1.

    Returns: 
    eigenvalues : ndarray 
        The eigenvalues of the Laplacian matrix.
    
    eigenvectors : ndarray 
        The eigenvectors corresponding to the eigenvalues.

    index_grid : ndarray 
        2D array with indices starting from 0 for points inside the circle, and None 
        for points outside.
    """
    mask, index_grid = initialize_grid(L,N)
    num_circle_points = np.count_nonzero(mask)
    laplacian = sp.lil_matrix((num_circle_points, num_circle_points), dtype= int)
    for i in range(N): 
        for j in range(N): 
            if index_grid[i, j] != None:
                idx = index_grid[i,j]
                laplacian[idx, idx] = -4
                for di, dj in  [(-1, 0), (1, 0), (0, -1), (0, 1)]: 
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < N and index_grid[ni, nj] != None:
                        laplacian[idx, index_grid[ni, nj]] = 1
    
    eigenvalues, eigenvectors = la.eigh(laplacian.toarray())

    return eigenvalues, eigenvectors, index_grid

def plot_eigenvectors(L, N, k): 

    eigenvalues, eigenvectors, index_grid = solve_circle_laplacian(L, N)
    _, axes = plt.subplots(2, k//2, figsize=(6, 6))

    sorted_eigenvalue_indices = np.argsort(np.abs(eigenvalues))

    for i, ax in enumerate(axes.flat):
        mode_values = np.zeros((N, N))  
        mode_values[index_grid != None] = eigenvectors[:, sorted_eigenvalue_indices[i]]
        ax.imshow(mode_values, extent=(-L/2, L/2, -L/2, L/2), origin="lower", cmap="bwr")
        ax.set_title(f"λ={eigenvalues[sorted_eigenvalue_indices[i]]:.4f}")

    plt.tight_layout()
    plt.show()

plot_eigenvectors(L = 1, N = 50, k = 6)

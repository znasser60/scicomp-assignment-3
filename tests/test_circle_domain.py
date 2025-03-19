import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from scicomp.eig_val_calc.circle import (
    initialize_grid,
    solve_circle_diffusion,
    construct_circle_laplacian,
)


@given(
    L=st.floats(min_value=1, max_value=10), 
    N=st.integers(min_value=10, max_value=100)
)
def test_circle_grid_initialization(L, N):
    mask, index_grid = initialize_grid(L, N)

    assert np.isnan(index_grid[~mask]).all(), "Non-masked values should be NaN"
    assert index_grid[mask].size == np.count_nonzero(mask), "Index grid mask mismatch"
    assert index_grid.size == N**2, "Grid size mismatch"
    assert (index_grid[mask] >= 0).all(), "Index grid values should be non-negative"


def test_laplacian_sparsity():
    """Ensure that each row of the Laplacian matrix has at most 5 nonzero values."""
    length = 4
    n = 150
    _, index_grid = initialize_grid(length, n)
    laplacian = construct_circle_laplacian(index_grid, length, n, use_sparse=True)
    
    
    row_counts = [len(row) for row in laplacian.rows]

    assert np.all(np.array(row_counts) <= 5), "Some rows have more than 5 nonzero entries"


def test_concenctration_grid_values():
    """Test the steady-state diffusion solution behaves correctly."""
    length, n = 10, 50
    source_position = (0, 0)
    
    c_grid = solve_circle_diffusion(source_position, length, n)
    
    assert np.nanmax(c_grid) <= 1.0, "Concentration should not exceed 1"
    assert np.nanmin(c_grid) >= 0.0, "Concentration should not be negative"


def test_diffusion_source_outside_circle():
    """Check that an error is raised when source is placed outside 
    the circle."""
    length, n = 10, 50
    source_position = (100, 100)  
    
    with pytest.raises(ValueError):
        solve_circle_diffusion(source_position, length, n)

if __name__ == "__main__":
    pytest.main([ __file__])


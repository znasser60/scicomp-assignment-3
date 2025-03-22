"""Functions that determine the linear algebraic solvers."""

from collections.abc import Callable
from functools import partial

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sp_la


def select_eig_solver(
    ev_length: int,
    k: int,
    use_sparse: bool = False,
    shift_invert: bool = False,
) -> Callable:
    """Sets up the eigenvalue solver function.

    Args:
        ev_length: Number of elements in solution eigenvectors.
        k: Number of eigenvalues to compute.
        use_sparse: Boolean to use sparse solver of not.
        shift_invert: Boolean to use shift inverse or not.

    Returns:
        Solver function.
    """
    match use_sparse, shift_invert:
        case False, _:
            eig_solver = la.eigh
        case True, True:
            eig_solver = partial(sp_la.eigsh, k=k, sigma=0, v0=np.ones(ev_length))
        case True, False:
            eig_solver = partial(sp_la.eigsh, k=k, which="SM", v0=np.ones(ev_length))

    return eig_solver


def select_diff_solver(use_sparse: bool) -> Callable:
    """Determines the solver for the linear diffusion system.

    Args:
        use_sparse: Boolean to use sparse solver of not.

    Returns:
        Linear algebraic solver function.
    """
    diff_solver = sp_la.spsolve if use_sparse else la.solve

    return diff_solver

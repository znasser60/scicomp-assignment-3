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
    use_eigsh: bool = True,
    shift_invert: bool = False,
) -> Callable:
    """Sets up the eigenvalue solver function.

    Args:
        ev_length: Number of elements in solution eigenvectors.
        k: Number of eigenvalues to compute.
        use_sparse: Boolean to use sparse solver of not.
        use_eigsh: Use the eigsh eigensolver (applicable for sparse eigensolvers).
        shift_invert: Boolean to use shift inverse or not.

    Returns:
        Solver function.
    """
    if not use_sparse:
        eig_solver = la.eigh
    else:
        solver_fn = sp_la.eigsh if use_eigsh else sp_la.eigs
        if shift_invert:
            eig_solver = partial(solver_fn, k=k, sigma=0, v0=np.ones(ev_length))
        else:
            eig_solver = partial(solver_fn, k=k, which="SM", v0=np.ones(ev_length))

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

"""Package for direct solutions to wave/diffusion equations, and leapfrog simulation.

This package provides low-level functionality and solvers for the 2D wave equation
with eigensolutions, direct methods for solving the time-independent diffusion equation,
and higher-order accuracy time-discretisation of spring systems through the leapfrog
method.

A command-line interface is additionally included for convenient
access to these simulations.
"""

from .utils import configure_mpl

configure_mpl()

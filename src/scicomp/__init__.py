"""Package for solving eigenmodes and diffusion.

This package provides functionality for solving eigenmodes of 2D membranes,
including various shapes such as squares, rectangles, and circles. It also
includes methods for solving steady-state diffusion equations using both
direct and iterative methods.

A command-line interface is included for easy access to these simulations.
"""

from .utils import configure_mpl

configure_mpl()

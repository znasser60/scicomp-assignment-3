"""Plotting examples."""

from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

from scicomp.shape import Shape
from scicomp.visualisation import (
    plot_circular_state,
    plot_eigenmode,
    plot_rectangular_state,
)


def plot_grid_values():
    """Example usage of functions to plot rect and circular domains."""
    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    # ===============================
    # Plot rectangle
    # ===============================
    frac_width = Fraction(6)
    frac_height = Fraction(9)
    nx: int = 1000
    dx = frac_width / nx
    ny = frac_height / dx

    assert ny.is_integer()

    width = float(frac_width)
    height = float(frac_height)
    ny = ny.numerator

    X, Y = np.meshgrid(np.linspace(0, width, nx + 1), np.linspace(0, height, ny + 1))
    state = np.sin(X) + np.cos(Y)
    state[[0, -1]] = 0
    state[:, [0, -1]] = 0
    plot_rectangular_state(state, width, height, axes[0])

    # ===============================
    # Plot circle
    # ===============================
    frac_diameter = Fraction(6)
    nx: int = 1000
    dx = frac_diameter / nx

    diameter = float(frac_diameter)
    radius = diameter / 2
    X, Y = np.meshgrid(
        np.linspace(-radius, radius, nx + 1), np.linspace(-radius, radius, nx + 1)
    )
    mask = radius**2 > X**2 + Y**2
    state = np.sin(X) + np.cos(Y)
    state[~mask] = 0
    plot_circular_state(state, diameter, nx, ax=axes[1])

    plt.show()


def plot_eigenmodes():
    """Plot some fake eigenmodes labelled by their eigenfrequencies."""
    fig, axes = plt.subplots(1, 4, sharey=True, constrained_layout=True)

    # Make grid roughly 4pi in each dimension
    frac_width = Fraction(88 / 7)
    frac_height = Fraction(88 / 7)
    nx: int = 1000
    dx = frac_width / nx
    ny = frac_height / dx

    assert ny.is_integer()

    width = float(frac_width)
    height = float(frac_height)
    ny = ny.numerator

    # Mock up some fake eigenthing results
    X, Y = np.meshgrid(np.linspace(0, width, nx + 1), np.linspace(0, height, ny + 1))
    eigenfrequencies = np.array([1 / 4, 1 / 2, 1.0, 2.0])
    eigenmodes = np.sin(eigenfrequencies[:, None, None] * X[None, ...]) + np.cos(
        eigenfrequencies[:, None, None] * Y[None, ...]
    )
    eigenmodes[:, [0, -1]] = 0
    eigenmodes[:, :, [0, -1]] = 0

    # Reshape into state vector, as this is the expected input
    eigenmodes = np.reshape(eigenmodes, (4, -1))
    for i, ef in enumerate(eigenfrequencies):
        plot_eigenmode(
            eigenmodes[i], ef, Shape.Square, axes[i], width=width, height=height
        )

    plt.show()


if __name__ == "__main__":
    plot_grid_values()
    plot_eigenmodes()

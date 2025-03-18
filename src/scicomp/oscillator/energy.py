"""Calculating energy of a 1D spring."""

from typing import TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", float, npt.NDArray[np.float64])


def calculate_spring_energy(x: T, v: T, k: float | int, m: float | int) -> T:
    """Calculate the instataneous energy in a 1D spring.

    Assuming no gravitational forces are acting, the energy of the spring is
    defined by the sum of the instantaneous elastic potential energy and
    kinetic energy.

    Args:
        x: A float or numpy array representing the displacement of
            the spring mass (or a sequence of displacements).
        v: A float or numpy array representing the velocity of
            an object (or a sequence of velocities).
        k: Spring constant (N/m).
        m: The mass of the object (kg).

    Returns:
        The total energy experience by the spring at each specified
        displacement and velocity.
        The return type matches that of the input displacements and velocities.
    """
    elastic_potential_energy = calculate_elastic_potential_energy(x, k)
    kinetic_energy = calculate_kinetic_energy(v, m)
    return elastic_potential_energy + kinetic_energy


def calculate_elastic_potential_energy(x: T, k: float | int) -> T:
    """Compute the elastic potential energy in a 1D spring.

    This is defined by the integral of the force experienced by
    a spring when extended to position $x$:

        U(x) = 1/2 kx^2

    Args:
        x: A float or numpy array representing the displacement of
            the spring mass (or a sequence of displacements).
        k: Spring constant (N/m).

    Returns:
        The elastic potential energy of the spring at each specified
        displacement.
        The return type is the same as the input type of argument `x`.

    Raises:
        ValueError if k < 1e-10 N/m. We assume nonzero k, and small values can produce
            unexpected behaviour.
        ValueError if k > 1e20 N/m. Large values of k can cause overflow due to
            multiplication by x^2.
        ValueError if maximum absolute displacement > 1e20 m. Large absolute
            displacements can cause overflow due to the square operation.
    """
    if not (1e-10 <= k <= 1e20):
        raise ValueError(
            f"Spring constant `k` must be in interval [1e-10 N/m, 1e20 N/m], "
            f"found {k} N/m."
        )

    max_abs_x = np.abs(x).max() if isinstance(x, np.ndarray) else abs(x)
    if max_abs_x > 1e20:
        raise ValueError(
            f"Displacement `x` must be in interval [-1e20m, 1e20m], found {x}m."
        )

    return (1 / 2) * k * x**2


def calculate_kinetic_energy(v: T, m: float | int) -> T:
    """Compute the kinetic energy of an object from its mass and velocity.

    The kinetic energy is defined by:

        U(x) = 1/2 mv^2

    Args:
        v: A float or numpy array representing the velocity of
            an object (or a sequence of velocities).
        m: The mass of the object (kg).

    Returns:
        The kinetic energy of the object at each specified velocity.
        The return type is the same as the input type of argument `x`.

    Raises:
        ValueError if m < 1e-10 kg. We assume nonzero mass, and small values can produce
            unexpected behaviour.
        ValueError if m > 1e20 kg. Large values of m can cause overflow due to
            multiplication by v^2.
        ValueError if maximum speed > 1e20 m/s. Large values can cause overflow due to
            the square operation when calculating kinetic energy.
    """
    if not (1e-10 <= m <= 1e20):
        raise ValueError(
            f"Mass `m` must be in interval [1e-10 kg, 1e20 km], found {m} kg."
        )

    max_abs_v = np.abs(v).max() if isinstance(v, np.ndarray) else abs(v)
    if max_abs_v > 1e20:
        raise ValueError(
            f"Velocity `v` must be in interval [-1e20 m/s, 1e20 m/s], found {v} m/s."
        )

    return (1 / 2) * m * v**2

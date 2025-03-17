"""Leapfrog method to simulate a 1D harmonic oscillator."""

from fractions import Fraction

import numpy as np
import numpy.typing as npt


def simulate_oscillator(
    x0: float,
    v0: float,
    k: float,
    m: float,
    dt: Fraction,
    runtime: int | Fraction,
) -> npt.NDArray[np.float64]:
    """Simulate a spring under simple harmonic motion with Leapfrog method.

    Args:
        x0: Initial position.
        v0: Initial velocity.
        k: Spring constant (N/m).
        m: Mass of object at the end of the spring (kg).
        dt: Duration of a discrete time-step.
        runtime: Total duration to simulate.

    Returns:
        Numpy array with shape (time, 2), where the first column contains
        velocity measurements, and the second contains position measurements.
    """
    n_steps_frac = runtime / dt
    if not n_steps_frac.is_integer():
        raise ValueError("`dt` does not divide `runtime`.")
    else:
        delta_t = float(dt)
        n_steps = int(n_steps_frac)

    # Precompute squared angular frequency
    omega_sq = k / m

    # Calculate v_(1/2) and x_1 separately first
    vhalf = update_velocity(v0, x0, omega_sq, float(delta_t / 2))
    x1 = update_position(x0, vhalf, float(delta_t))

    # Iterate until n == n_steps, recording new x and halfstep v for each n
    states = np.empty((n_steps + 1, 2), dtype=np.float64)
    states[0] = (v0, x0)
    states[1] = (vhalf, x1)
    for n in range(2, n_steps + 1):
        states[n] = update_state(*states[n - 1], omega_sq=omega_sq, dt=delta_t)

    return states


def update_state(
    v_prev: float, x_prev: float, omega_sq: float, dt: float
) -> tuple[float, float]:
    """Perform a single update of the discrete 1D spring scheme.

    Args:
        v_prev: Velocity at the previous time-step.
        x_prev: Position at the previous time-step.
        omega_sq: Squared angular frequency of the spring system.
        dt: Discrete time-step duration.

    Returns:
        Tuple containing the new velocity and new position, both as floats.
    """
    v_new = update_velocity(v_prev, x_prev, omega_sq, dt)
    x_new = update_position(x_prev, v_new, dt)
    return (v_new, x_new)


def update_velocity(v_prev: float, x_prev: float, omega_sq: float, dt: float) -> float:
    """Calculate updated velocity of the discrete 1D spring scheme.

    Args:
        v_prev: Previous velocity.
        x_prev: Previous position.
        omega_sq: Squared angular frequency of the spring system.
        dt: Discrete time-step duration.

    Returns:
        New velocity at the next time-step.
    """
    return v_prev - dt * omega_sq * x_prev


def update_position(x_prev: float, v_prev: float, dt: float) -> float:
    """Calculate updated position of the discrete 1D spring scheme.

    Args:
        x_prev: Previous position.
        v_prev: Previous velocity.
        dt: Discrete time-step duration.

    Returns:
        New position at the next time-step.
    """
    return x_prev + dt * v_prev

"""Simulation of 1D harmonic oscillator using Runge-Kutta 4/5."""

from fractions import Fraction

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp


def oscillator_derivatives(_t, y, k: float, m: float) -> tuple[float, float]:
    """1D harmonic oscillator position and velocity deriviatives.

    For use in scipy.integrate.solve_ivp

    Args:
        _t: (Unused) current simulation time.
        y: Tuple (x, v) position and velocity at time = `t`
        k: Spring constant (N/m)
        m: Mass at end of spring (kg)

    Returns:
        Tuple containing the time derivatives of position and velocity
        at time = `t`.
    """
    x_i, v_i = y
    dx_dt = v_i
    dv_dt = -k * x_i / m
    return (dx_dt, dv_dt)


def simulate_oscillator(
    x0: float,
    v0: float,
    k: float,
    m: float,
    dt: Fraction,
    runtime: int | Fraction,
) -> npt.NDArray[np.float64]:
    """Simulate a spring under simple harmonic motion with Runge-Kutta 4/5.

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
        n_steps = int(n_steps_frac)
        t_eval = np.linspace(0, float(runtime), n_steps + 1)

    t_span = (0, runtime)
    y0 = (x0, v0)
    solution = solve_ivp(
        oscillator_derivatives,
        t_span=t_span,
        y0=y0,
        method="RK45",
        t_eval=t_eval,
        args=(k, m),
    )

    return solution.y.T

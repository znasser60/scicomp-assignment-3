"""Frequency, eigenmod calculators and u(x, y, t) = v(x, y)T(t) evaluator."""

import numpy as np
import numpy.typing as npt


def eval_oscillating_solution(
    t: float | int,
    mode: npt.NDArray[np.float64],
    freq: float,
    c: float = 1.0,
    cosine_coef: float | int = 1.0,
    sine_coef: float | int = 1.0,
) -> npt.NDArray[np.float64]:
    """Evaluate an eigensolution of the form K < 0 at a particular time.

    Args:
        t: Time-point to evaluate.
        mode: Eigenmode as a vector.
        freq: Eigenfrequency (sqrt(-K)) associated with the eigenmode.
        c: Wave propagation velocity (m/s).
        cosine_coef: Multiplication coefficient for the cosine term in T(t).
        sine_coef: Multiplication coefficient for the sine term in T(t).

    Returns:
        Amplitude at time t, given the input eigenmode and eigenfrequency.
        The returned vector has the same shape as the mode parameter.
    """
    if freq <= 0:
        raise ValueError(f"Expected eigenfrequency λ > 0, received {freq:.4f}")

    cosine_component = cosine_coef * np.cos(c * freq * t)
    sine_component = sine_coef * np.sin(c * freq * t)
    return mode * (cosine_component + sine_component)

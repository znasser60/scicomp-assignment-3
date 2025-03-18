from fractions import Fraction

import pytest
from hypothesis import assume, given
from hypothesis.strategies import floats, fractions, integers

from scicomp.oscillator.leapfrog import (
    is_leapfrog_oscillator_stable,
    simulate_oscillator,
)


@given(
    x0=floats(min_value=-1e20, max_value=1e20),
    v0=floats(min_value=-1e20, max_value=1e20),
    k=floats(min_value=1e-5, max_value=1e4),
    omega=floats(min_value=1e-5, max_value=2),
    dt=fractions(min_value=1e-10, max_value=1),
    n_iterations=integers(min_value=1, max_value=10000),
)
def test_simulate_oscillator_params(x0, v0, k, omega, dt, n_iterations):
    """Verify that `simulate_oscillator` runs smoothly for valid params."""
    assume(dt < omega / 2)
    runtime = n_iterations * dt
    m = k / omega**2
    assume(m <= 1e5)
    if runtime.denominator == 1:
        runtime = runtime.numerator

    simulate_oscillator(x0, v0, k, m, dt, runtime)


def test_simulate_oscillator_raises_for_dt_not_divide_runtime():
    x0 = 10
    v0 = 0
    k = 2
    m = 1
    dt = Fraction(1, 2)
    runtime = Fraction(16, 10)

    with pytest.raises(ValueError):
        simulate_oscillator(x0, v0, k, m, dt, runtime)


def test_simulate_oscillator_raises_for_dt_too_small():
    x0 = 10
    v0 = 0
    k = 1
    m = 1
    dt = Fraction(1, 100_000_000_000)
    runtime = 1

    with pytest.raises(ValueError):
        simulate_oscillator(x0, v0, k, m, dt, runtime)


def test_simulate_oscillator_raises_for_dt_too_large():
    x0 = 10
    v0 = 0
    k = 1000
    m = 1
    dt = Fraction(11, 1)
    runtime = 22

    with pytest.raises(ValueError):
        simulate_oscillator(x0, v0, k, m, dt, runtime)


def test_stability_false_for_unstable_dt_on_boundary():
    omega = 1.0
    dt = 0.5
    stable = is_leapfrog_oscillator_stable(dt, omega)
    assert not stable


def test_stability_false_for_very_unstable_dt():
    omega = 1.0
    dt = 2
    stable = is_leapfrog_oscillator_stable(dt, omega)
    assert not stable


def test_stability_true_for_just_stable_dt():
    omega = 1.0
    dt = 0.49
    stable = is_leapfrog_oscillator_stable(dt, omega)
    assert stable


def test_stability_true_for_stable_dt():
    omega = 1.0
    dt = 0.25
    stable = is_leapfrog_oscillator_stable(dt, omega)
    assert stable


def test_simulate_oscillator_raises_for_unstable_dt():
    x0 = 10
    v0 = 0
    k = 1
    m = 1

    # For k=1, m=1, omega = sqrt(k/m) = 1. Set dt to be no less than this.
    dt = Fraction(1, 2)
    runtime = 1

    with pytest.raises(ValueError):
        simulate_oscillator(x0, v0, k, m, dt, runtime)

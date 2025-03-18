import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.strategies import floats

from scicomp.oscillator import energy


def test_kinetic_energy_zero_for_stationary():
    m = 1
    v = 0
    e_kinetic = energy.calculate_kinetic_energy(v, m)
    assert math.isclose(e_kinetic, 0.0)


def test_kinetic_energy_zero_for_stationary_vector():
    m = 1
    v = np.zeros(100, dtype=np.float64)
    e_kinetic = energy.calculate_kinetic_energy(v, m)
    assert np.allclose(e_kinetic, 0.0)


def test_elastic_energy_zero_for_no_displacement():
    k = 1
    x = 0
    e_elastic = energy.calculate_elastic_potential_energy(x, k)
    assert math.isclose(e_elastic, 0.0)


def test_elastic_energy_zero_for_no_displacement_vector():
    k = 1
    x = np.zeros(100, dtype=np.float64)
    e_elastic = energy.calculate_elastic_potential_energy(x, k)
    assert np.allclose(e_elastic, 0.0)


def test_spring_energy_zero_for_stationary_no_displacement():
    k = 1
    m = 1
    x = 0
    v = 0
    e_spring = energy.calculate_spring_energy(x, v, k, m)
    assert math.isclose(e_spring, 0.0)


def test_spring_energy_zero_for_stationary_no_displacement_vector():
    k = 1
    m = 1
    x = np.zeros(100, dtype=np.float64)
    v = np.zeros(100, dtype=np.float64)
    e_spring = energy.calculate_spring_energy(x, v, k, m)
    assert np.allclose(e_spring, 0.0)


@given(
    v=floats(min_value=-1e20, max_value=1e20),
    m=floats(min_value=1e-10, max_value=1e20),
)
def test_kinetic_energy_nonzero_for_moving(v, m):
    # Assume that v sufficiently far from zero to avoid underflow.
    assume(abs(v) > 1e-20)
    e_kinetic = energy.calculate_kinetic_energy(v, m)
    assert e_kinetic > 0


@given(
    x=floats(min_value=-1e20, max_value=1e20),
    k=floats(min_value=1e-10, max_value=1e20),
)
def test_elastic_energy_nonzero_for_nonzero_displacement(x, k):
    # Assume that x sufficiently far from zero to avoid underflow.
    assume(abs(x) > 1e-20)
    e_elastic = energy.calculate_elastic_potential_energy(x, k)
    assert e_elastic > 0


def test_kinetic_energy_raises_on_m_too_large():
    v = 10
    m = 1e21
    with pytest.raises(ValueError):
        energy.calculate_kinetic_energy(v, m)


def test_kinetic_energy_raises_on_v_too_large_negative():
    v = -1e21
    m = 1
    with pytest.raises(ValueError):
        energy.calculate_kinetic_energy(v, m)


def test_kinetic_energy_raises_on_v_too_large():
    v = 1e21
    m = 1
    with pytest.raises(ValueError):
        energy.calculate_kinetic_energy(v, m)


def test_elastic_energy_raises_on_k_too_small():
    x = 10
    k = 9e-11
    with pytest.raises(ValueError):
        energy.calculate_elastic_potential_energy(x, k)


def test_elastic_energy_raises_on_k_too_large():
    x = 10
    k = 1e21
    with pytest.raises(ValueError):
        energy.calculate_elastic_potential_energy(x, k)


def test_elastic_energy_raises_on_x_too_large_negative():
    x = -1e21
    k = 1
    with pytest.raises(ValueError):
        energy.calculate_elastic_potential_energy(x, k)


def test_elastic_energy_raises_on_x_too_large():
    x = 1e21
    k = 1
    with pytest.raises(ValueError):
        energy.calculate_elastic_potential_energy(x, k)


@given(
    abs_v=floats(min_value=-1e20, max_value=1e20),
    m=floats(min_value=1e-10, max_value=1e20),
)
def test_kinetic_energy_direction_invariant(abs_v, m):
    """Elastic potential energy should be invariant to the displacement direction."""
    e_kinetic_pos = energy.calculate_kinetic_energy(abs_v, m)
    e_kinetic_neg = energy.calculate_kinetic_energy(-abs_v, m)
    assert math.isclose(e_kinetic_pos, e_kinetic_neg)


@given(
    abs_x=floats(min_value=-1e20, max_value=1e20),
    k=floats(min_value=1e-10, max_value=1e20),
)
def test_elastic_energy_direction_invariant(abs_x, k):
    """Elastic potential energy should be invariant to the displacement direction."""
    e_elastic_pos = energy.calculate_elastic_potential_energy(abs_x, k)
    e_elastic_neg = energy.calculate_elastic_potential_energy(-abs_x, k)
    assert math.isclose(e_elastic_pos, e_elastic_neg)


def test_conservation_of_energy_in_spring():
    """From conservation of energy, total energy should be constant.

    This test verifies that the total energy calculated when the spring
    is at maximum extension and zero velocity matches the total energy
    when the spring is at zero extension, but maximum velocity (i.e.,
    passing the fixed point).
    """
    k = 2
    m = 2

    # Measurement 1: spring at maximum extension
    x0 = 1
    v0 = 0
    e0 = energy.calculate_spring_energy(x0, v0, k, m)

    # Measurement 2: spring at maximum velocity
    x1 = 0
    v1 = -1
    e1 = energy.calculate_spring_energy(x1, v1, k, m)

    # Verify conservation
    assert math.isclose(e0, e1)

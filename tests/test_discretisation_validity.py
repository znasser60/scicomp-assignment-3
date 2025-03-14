from fractions import Fraction

from hypothesis import given
from hypothesis.strategies import fractions, integers, one_of

from scicomp.utils.verification import check_valid_spatial_discretisation


@given(
    width=one_of(
        integers(min_value=1),
        fractions(min_value=1e-6),
    ),
    n=integers(min_value=1),
)
def test_square_discretisation_always_valid_for_pos_n(width, n):
    assert check_valid_spatial_discretisation(width=width, height=width, n=n)


@given(
    width=one_of(
        integers(min_value=1),
        fractions(min_value=1e-6),
    ),
    height_mult=integers(min_value=1),
    n=integers(min_value=1),
)
def test_rect_discretisation_always_valid_for_height_multiple_of_width(
    width, height_mult, n
):
    assert check_valid_spatial_discretisation(
        width=width, height=height_mult * width, n=n
    )


def test_valid_discretisation_good_gcd():
    """Scheme should be valid because dx=w/n divides both w and h."""
    width = Fraction(3, 2)  # 1.5
    height = Fraction(5, 2)  # 2.5
    n = 3  # dx = 0.5, divides 1.5 and 2.5
    assert check_valid_spatial_discretisation(width, height, n)


def test_invalid_discretisation_bad_gcd():
    """Scheme should be invalid because dx=w/n=1/2 does not divide 1.6"""
    width = 1
    height = Fraction(16, 10)
    n = 2
    assert not check_valid_spatial_discretisation(width, height, n)


def test_invalid_discretisation_dx_larger_than_height():
    """Scheme should be invalid because dx=w/n is larger than height."""
    width = 6
    height = Fraction(3, 2)
    n = 2
    assert not check_valid_spatial_discretisation(width, height, n)

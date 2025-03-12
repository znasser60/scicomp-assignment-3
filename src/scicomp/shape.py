"""Define the shapes which are implemented for simulation."""

import math
from enum import StrEnum
from fractions import Fraction

import numpy as np
import numpy.typing as npt


class Shape(StrEnum):
    """Shapes implemented for simulation."""

    Circle = "circle"
    Square = "square"
    Rectangle = "rectangle"

    def state_vec_to_grid(
        self, state_vector: npt.NDArray[np.float64], n: int | None = None
    ) -> npt.NDArray[np.float64]:
        """Convert a state vector into the grid representation of associated shape."""
        if state_vector.ndim != 1:
            raise ValueError(
                f"Expected state_vector with 1 dimension, but received argument "
                f"has {state_vector.ndim}. Huh?"
            )
        match self:
            case Shape.Square | Shape.Circle:
                n = int(math.sqrt(state_vector.size))
                if (n**2) != state_vector.size:
                    raise TypeError(
                        f"Invalid shape for {self} state_vector. Size is not a "
                        "perfect square."
                    )
                grid = np.reshape(state_vector, (n, n))
            case Shape.Rectangle:
                if n is None:
                    raise ValueError("Must supply n to reshape type Shape.Rectangle.")
                ny = Fraction(state_vector.size, n)
                if not ny.is_integer():
                    raise TypeError(
                        f"Invalid shape for rectangle state_vector with {n=}, "
                        "n does not divide state_vector.size"
                    )
                grid = np.reshape(state_vector, (ny.numerator, n))
        return grid

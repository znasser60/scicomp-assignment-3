"""Define shapes and functionality."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from scicomp.solvers import select_diff_solver as select_diffusion_solver
from scicomp.solvers import select_eig_solver as select_eigenproblem_solver
from scicomp.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class ShapeEnum(StrEnum):
    """Enum corresponding to the different defined Domains."""

    Rectangle = "rectangle"
    Square = "square"
    Circle = "circle"

    def domain(
        self, width: int | Fraction, height: int | Fraction | None = None
    ) -> Domain:
        """Construct a domain with the given shape."""
        if height is not None and self != ShapeEnum.Rectangle:
            logger.warning(
                "`height` parameter has no effect for shapes other than Rectangle."
            )
        elif height is None and self == ShapeEnum.Rectangle:
            logger.warning(
                "No `height` parameter provided for shape `Rectangle`, "
                "re-using `width`."
            )
            height = width

        match self:
            case ShapeEnum.Square:
                d = Rectangle(width)
            case ShapeEnum.Rectangle:
                d = Rectangle(width, height)
            case ShapeEnum.Circle:
                d = Circle(width)

        return d


class Domain(ABC):
    """Shape domains to use for eigenmode and diffusion simulations."""

    _length: Fraction

    def __init__(self, length: int | Fraction):
        """Construct domain."""
        self._length = Fraction(length)

    def solve_eigenproblem(
        self,
        k: int,
        ny: int,
        use_sparse: bool = True,
        shift_invert: bool = True,
        index_grid: npt.NDArray[np.float64] | None = None,
        laplacian: npt.NDArray[np.float64] | sp.lil_matrix | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculates the frequencies and the eigenmodes.

        The equation the function solves is assumed to have the form: Mv = Kv,
        where M is the laplacian matrix and -K is the ω/c^2.

        Args:
            k: Number of eigenvalues to compute. If `use_sparse` is True, this must
                be less than (ny+1) x (nx + 1).
            ny: Resolution of the discretisation in y-axis.
            use_sparse: Boolean to use sparse solver of not.
            shift_invert: Boolean to use shift inverse or not.
            index_grid: Optional pre-discretised index grid.
            laplacian: Optional pre-computed discrete laplacian matrix.

        Returns:
            Frequencies (ω) (1D numpy array) and eigenmodes (v) (2D numpy array).
        """
        if laplacian is None:
            laplacian = self.construct_discrete_laplacian(
                ny=ny,
                use_sparse=use_sparse,
                divide_stepsize=False,
                index_grid=index_grid,
            )
        else:
            use_sparse = isinstance(laplacian, sp.lil_matrix)

        solver = select_eigenproblem_solver(
            ev_length=self.discretisation_size(ny=ny, index_grid=index_grid),
            k=k,
            use_sparse=use_sparse,
            shift_invert=shift_invert,
        )

        eigenvalues, eigenvectors = solver(laplacian)

        # Eigenfrequency is sqrt(-λ)/h = sqrt(-λ) * n / length..
        eigenfrequencies = ((-eigenvalues) ** 0.5) * (ny / self.length)

        # Determine how many of the should be returned
        sort_idx = np.argsort(np.abs(eigenvalues))[:k]

        return eigenfrequencies[sort_idx], eigenvectors[:, sort_idx]

    def construct_discrete_laplacian(
        self,
        ny: int | None = None,
        use_sparse: bool = False,
        divide_stepsize: bool = True,
        index_grid: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Construct the discrete Laplacian matrix for domain.

        Args:
            ny: Resolution of the discretisation in y-axis.
            use_sparse: Boolean to use sparse solver of not.
            divide_stepsize: Boolean, if true, divides Laplacian by h^2.
            index_grid: Optional pre-discretised index grid.

        Returns:
            Discrete Laplacian matrix of size MxM where M = (ny + 1) x (nx + 1).
        """
        if index_grid is None:
            if ny is None:
                raise ValueError("Exactly one of `ny`, `index_grid` must be provided.")
            index_grid = self.discretise(ny)
        if ny is None:
            ny = index_grid.shape[0] - 1
        nx = index_grid.shape[1] - 1
        assert ny is not None

        n_points = self.discretisation_size(index_grid=index_grid)
        matrix_type = sp.lil_matrix if use_sparse else np.zeros
        laplacian = matrix_type((n_points, n_points), dtype=np.float64)
        for i in range(ny + 1):
            for j in range(nx + 1):
                # Ensure the point is part of the shape
                if np.isnan(index_grid[i, j]):
                    continue
                idx = int(index_grid[i, j])

                # Fill the main diagonal
                laplacian[idx, idx] = -4

                # Find the fill the neighbour values in the row of the corresp. cell
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj

                    # Check if the neighbour exists (if not bound condition applied)
                    if (
                        0 <= ni < ny + 1
                        and 0 <= nj < nx + 1
                        and not np.isnan(index_grid[ni, nj])
                    ):
                        laplacian[idx, int(index_grid[ni, nj])] = 1

        if divide_stepsize:
            h = self.calculate_step_size(ny)
            laplacian /= float(h**2)

        return laplacian

    def discretisation_size(
        self, ny: int | None = None, index_grid: npt.NDArray[np.float64] | None = None
    ) -> int:
        """Calculate the number of elements in the discretised matrix.

        Exactly one of `ny` and `index_grid` must be supplied.

        Args:
            ny: Resolution of the discretization of the y-axis.
            index_grid: Pre-computed discretised index grid.

        Returns:
            Integer number of elements.
        """
        if index_grid is None and ny is None:
            raise ValueError("Exactly one of `ny`, `index_grid` must be provided.")
        elif index_grid is None and ny is not None:
            index_grid = self.discretise(ny)
        assert index_grid is not None
        return (~np.isnan(index_grid)).sum()

    def discretise(self, ny: int) -> npt.NDArray[np.float64]:
        """Discretise the domain into a grid of indices.

        This function creates a square (N + 1) x (N + 1) grid representing a domain
        with arbitrary shape, enclosed by a rectangle. The points inside the domain
        are assigned unique index values starting from 0, while points outside the
        circle are set to None.

        Args:
            ny: Resolution of the discretization of the y-axis.

        Returns:
            2D array with indices starting from 0 for points inside the circle, and None
            for points outside.
        """
        h = self.calculate_step_size(ny)
        nx = int(self.width / h)
        x_range = np.linspace(self.x_min, self.x_max, nx + 1)
        y_range = np.linspace(self.y_min, self.y_max, ny + 1)
        x, y = np.meshgrid(x_range, y_range)
        mask = self.contains(x, y)
        index_grid = np.full_like(x, np.nan, dtype=np.float64)
        index_grid[mask] = np.arange(np.sum(mask))

        logger.info("Indexing for domain has been created successfully.")

        return index_grid

    def calculate_step_size(self, n: int) -> Fraction:
        """Calculate discrete spatial step-size given n spatial intervals.

        Args:
            n: Resolution of discretisation.

        Returns:
            Fraction discrete spatial step size which divides the y-axis
            into `n` intervals.

        Raises:
            ValueError: If n cannot yield a valid scheme.
        """
        dy = Fraction(numerator=self.height, denominator=n)
        dx = self.width * (1 / dy)
        if dx.denominator != 1:
            raise ValueError(
                f"Number of spatial discretisation intervals {n=} invalid, cannot "
                f"divide shape with width={float(self.width)}, "
                f"height={float(self.height)} using fixed step-size."
            )
        return dy

    def solve_diffusion(
        self,
        source_position: tuple,
        ny: int,
        use_sparse: bool = False,
        laplacian: npt.NDArray[np.float64] | sp.lil_matrix | None = None,
    ):
        """Solves the steady state diffusion equation using direct methods (Mc = b).

        Args:
            source_position: Specified grid position of the source
            ny: Resolution of the discretisation in y-axis.
            use_sparse: Boolean to use sparse solver of not.
            laplacian: Optional pre-computed discrete laplacian matrix.

        Returns:
            A 2D array representing the concentration distribution across the grid
                on a circle domain.
        """
        index_grid = self.discretise(ny)
        if laplacian is None:
            laplacian = self.construct_discrete_laplacian(
                ny=None,
                use_sparse=use_sparse,
                divide_stepsize=False,
                index_grid=index_grid,
            )
        else:
            use_sparse = isinstance(laplacian, sp.lil_matrix)

        source_idx = self._calculate_source_idx(ny, source_position)

        # Update the original laplacian matrix with the source information
        laplacian[source_idx, :] = 0
        laplacian[source_idx, source_idx] = 1

        b = self._compute_diff_result_vector(ny, source_idx)

        solver = select_diffusion_solver(use_sparse)
        c = solver(laplacian.tocsr() if use_sparse else laplacian, b)

        c_grid = np.full((ny + 1, ny + 1), np.nan)
        c_grid[~np.isnan(index_grid)] = c

        return c_grid

    def _calculate_source_idx(self, ny: int, source_position: tuple) -> int:
        """Computes the index of the source.

        The index is determined in the c vector (flatten (1D) array of the original
        indexing given the shape) in the Mc = b system.

        Args:
            ny: Resolution of the discretisation in y-axis.
            source_position: Specified grid position of the source

        Returns:
            Source index.
        """
        x, y = (
            np.linspace(-self.length / 2, self.length / 2, ny),
            np.linspace(-self.length / 2, self.length / 2, ny),
        )

        index_grid = self.discretise(ny)
        source_idx = index_grid[
            np.argmin(np.abs(y - source_position[1])),
            np.argmin(np.abs(x - source_position[0])),
        ]

        return int(source_idx)

    def _compute_diff_result_vector(
        self, ny: int, source_idx: int
    ) -> npt.NDArray[np.float64]:
        """Computes the result vector in the Mc = b system.

        Args:
            ny: Resolution of the discretisation in y-axis.
            source_idx: Source index.

        Returns:
            'b' vector (1D numpy array)
        """
        index_grid = self.discretise(ny)
        n_in_shape_points = (~np.isnan(index_grid)).sum()
        b = np.zeros(n_in_shape_points)
        b[source_idx] = 1

        return b

    @abstractmethod
    def contains(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool]:
        """Create mask of x,y values contained within the shape."""
        ...

    @property
    def length(self) -> float:
        """Length of domain."""
        return float(self._length)

    @property
    def width(self) -> Fraction:
        """Physical domain width.

        Given that the shape's with and height are the same,
        otherwise it should be overridden in the shape specific domain class.
        """
        return self._length

    @property
    def height(self) -> Fraction:
        """Physical domain height.

        Given that the shape's with and height are the same,
        otherwise it should be overridden in the shape specific domain class.
        """
        return self._length

    @property
    @abstractmethod
    def x_min(self) -> float:
        """Minimum x-value contained within shape."""
        ...

    @property
    @abstractmethod
    def x_max(self) -> float:
        """Maximum x-value contained within shape."""
        ...

    @property
    @abstractmethod
    def y_min(self) -> float:
        """Minimum y-value contained within shape."""
        ...

    @property
    @abstractmethod
    def y_max(self) -> float:
        """Maximum y-value contained within shape."""
        ...


class Circle(Domain):
    """Circular domain, centered at the origin."""

    def __init__(self, diameter: int | Fraction):
        """Construct a circle."""
        super().__init__(diameter)
        self.diameter = diameter
        self.radius = diameter / 2

    def contains(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool]:
        """Create mask of x,y values contained within the circle."""
        return x**2 + y**2 < float(self.radius) ** 2

    @property
    def x_min(self) -> float:
        """Minimum x-value contained within circle."""
        return -self.length / 2

    @property
    def x_max(self) -> float:
        """Maximum x-value contained within circle."""
        return self.length / 2

    @property
    def y_min(self) -> float:
        """Minimum y-value contained within circle."""
        return -self.length / 2

    @property
    def y_max(self) -> float:
        """Maximum y-value contained within circle."""
        return self.length / 2


class Rectangle(Domain):
    """Rectangular domain, centered at the origin."""

    def __init__(self, a_side: int | Fraction, b_side: int | Fraction | None = None):
        """Initialisation of the Rectangle class.

        Arguments are going to set based on the following logic:
           ______________
          |             |
        b |             |
          |_____________|
                 a

        In case the b_side is not specified during the initialisation, a square is
        created with the side length equal a_side.
        """
        b_side = b_side or a_side
        super().__init__(max(a_side, b_side))
        self.a_side = Fraction(a_side)
        self.b_side = Fraction(b_side)

    def contains(
        self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool]:
        """Create a mask of x,y values contained within the rectangle."""
        return (np.abs(x) < float(self.a_side) / 2) & (
            np.abs(y) < float(self.b_side) / 2
        )

    @property
    def x_min(self) -> float:
        """Minimum x-value contained within rectangle."""
        return float(-self.a_side / 2)

    @property
    def x_max(self) -> float:
        """Maximum x-value contained within rectangle."""
        return float(self.a_side / 2)

    @property
    def y_min(self) -> float:
        """Minimum y-value contained within rectangle."""
        return float(-self.b_side / 2)

    @property
    def y_max(self) -> float:
        """Maximum y-value contained within rectangle."""
        return float(self.b_side / 2)

    @property
    def width(self) -> Fraction:
        """Physical domain width."""
        return self.a_side

    @property
    def height(self) -> Fraction:
        """Physical domain height."""
        return self.b_side

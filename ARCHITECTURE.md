# Code structure
The code in this repository falls into X categories:
- Source code (`src/scicomp`): Modular code intended for use as a Python package. Includes simulation/model code.
- Command-line interface (`src/scicomp/cli/*`): Defines low-level CLI access to simulations for animation and plotting purposes.
- Tests (`tests`).

Experiments are orchestrated by the [Makefile](Makefile), which internally uses the `scicomp` CLI to drive the 
simulations which form the basis of the results featured in the associated report.

## Source code
The majority of the code related to the eigenmodes and steady-state diffusion is found in `src/scicomp/model_elements/domains.py`. There
we define an abstract `Domain` base class from which the shapes used in these experiments inherit. As the systems in these 
experiments have very similar definitions, we define much of their shared functionality within `Domain`. This includes:

+ Discretisation validity checking (e.g., does the step-size divide both cartesian dimensions?)
+ Construction of the discrete Laplacian matrix
+ Masks used to select the cells contained within a shape from an enclosing rectangle
+ Various shape-or-plotting-related methods and properties, such as shape width, height, and axes-bounds.

New shapes are defined using the indicator function, f_Omega, from the associated report. In essence, to define 
a new shape, one is required to subclass the `Domain` abstract base class, and define the conditon under which a 
point (x,y) is contained within the shape.

The specific subroutines used to solve the problems in 3.1 and 3.2 are included as methods on the `Domain` class:

+ Identifying eigenmodes and eigenfrequencies: `Domain.solve_eigenproblem`
+ Solving a steady-state diffusion problem: `Domain.solve_steady_state_diffusion`

The remaining experiment-related code pertains to the harmonic oscillator simulation. This is located under `src/scicomp/oscillator`, which
contains three Python modules:

- `leapfrog.py`: Implements the numerical Leapfrog scheme for the simple harmonic oscillator. Includes functions to test for method 
    stability.
- `runge_kutta.py`: Provides a wrapper around `scipy.integrate.solve_ivp`, specialised for the harmonic oscillator problem. Exports a 
    function with the same method signature (though a different name) as is used to run the Leapfrog simulations. This simplifies
    comparison of the two methods.
- `energy.py`: Standalone functions for calculating the kinetic, elastic potential, and total energy of the harmonic oscillator system.

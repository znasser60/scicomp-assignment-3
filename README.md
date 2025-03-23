# Scientific Computing Assignment 3

Created for Assignment 3 of Scientific Computing 2025.

Simulating Eigenmodes on drums or membranes, steady-state diffusion on a circular domain, and a one-dimensional harmonic oscillator. 

## To contribute
See [contributing docs](CONTRIBUTING.md).

## Code architecture
See [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup 
To run this code, you will need to install uv to create the correct Python environment. 
Run: 
```bash
uv sync
```

## Running the code
All experiments are orchestrated by Make according to the [Makefile](Makefile). To
run the experiments, simply run:
```bash
$ make -j
```

To run an experiment that plots runtimes for different eigenvalue solvers, run: 
```bash
$ make serial
```

The resulting figures are located under `results/figures/`.

By default, experiments are run in a 'quick' mode with a coarser grid than is used 
in results from the associated report. You can use the `QUALITY` environment variable
to run the experiments with the parameters used in the report:
```bash
$ QUALITY=high make -j 
```

### Command-line interface
We also provide low-level command-line access to the simulations via the `scicomp` CLI. 
This can also be used to generate plots and animations included in this repository. 

To view all available commands (experiments), run:
```bash
$ uv run scicomp
```

To output plots: 
```bash
$ uv run scicomp plot
```

To run animations: 
```bash
$ uv run scicomp animate
```


import typer

from scicomp.eig_val_calc.circle import plot_circle_diffusion, solve_circle_diffusion

app = typer.Typer()


@app.command()
def circular_steady_state_diffusion(
    length: float = typer.Option(4.0, help="Diameter of the circular domain."),
    n: int = typer.Option(150, help="Number of grid points in each dimension."),
    source_position: tuple[float, float] = typer.Option(
        (0.6, 1.2), help="Position of the source on the grid."
    ),
):
    """Plots the steady-state diffusion solution on a circular domain."""
    c_grid = solve_circle_diffusion(source_position, length, n)
    plot_circle_diffusion(n, c_grid, length)

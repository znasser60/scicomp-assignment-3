"""ClI entrypoint"""

import typer

from .animations import app as animation_app
from .plots import app as plot_app
from .version import app as version_app

cli_desc = """
ASSIGNMENT 3 SHORT DESCRIPTION

The code implementation and all results are part of the solution for Assignment 3 in 
the Scientific Computing (2025) course at the Universiteit van Amsterdam.

The application is designed to calculate results in a concise, controlled, and 
reproducible manner.

[bold]Authors:[/] Zainab Nasser, Marcell Szegedi, and Henry Zwart
[bold]License:[/] This project is licensed under an \n
[link=https://opensource.org/license/MIT]MIT[/link] license.
"""

app = typer.Typer(
    rich_markup_mode="rich",
    help=cli_desc,
)

app.add_typer(version_app)
app.add_typer(plot_app, name="plot")
app.add_typer(animation_app, name="animation")

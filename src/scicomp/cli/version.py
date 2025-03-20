"""Creates the version command."""

import typer

from scicomp import __version__ as app_version

app = typer.Typer()


@app.command()
def version():
    """Specify the application version."""
    print(app_version)

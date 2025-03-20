"""CLI commands to animate the results."""

import typer

from .create_wave_animation import app as wave_anim_app

app = typer.Typer()

app.add_typer(wave_anim_app)

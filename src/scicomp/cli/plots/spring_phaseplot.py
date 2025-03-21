"""Plot position and velocity of 1D spring under simple harmonic motion."""

from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import typer

from scicomp.oscillator.leapfrog import simulate_oscillator

app = typer.Typer()


@app.command()
def spring_phaseplot():
    """Animate 1D spring."""
    m = 1
    x0 = 10
    v0 = 0
    dt = Fraction(1, 100)
    runtime = 5

    fig, ax = plt.subplots(figsize=(2.15, 2.25), constrained_layout=True)
    ax.set_xlabel("Displacement ($m$)")
    ax.set_ylabel("Velocity ($m/s$)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for k in (1, 3, 5):
        osc_period = 2 * np.pi * (m / k) ** 0.5
        runtime = Fraction(np.round(0.75 * osc_period, decimals=2)).limit_denominator(
            100
        )

        states = simulate_oscillator(x0, v0, k, m, dt, runtime)
        (lineplot,) = ax.plot(states[:, 1], states[:, 0], label=f"$k={k}$")
        ax.annotate(
            "",
            xy=(states[-1, 1] + 1, states[-1, 0]),
            xytext=(states[-1, 1], states[-1, 0]),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": lineplot.get_color()},
        )

    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8,
        handlelength=0.8,
        columnspacing=0.5,
        labelspacing=0.2,
        frameon=False,
    )

    fig.savefig("results/figures/spring_1d_phaseplot.png", bbox_inches="tight")
    fig.savefig("results/figures/spring_1d_phaseplot.pdf", bbox_inches="tight")

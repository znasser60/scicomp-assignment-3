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

    fig, axes = plt.subplots(
        1, 3, figsize=(3.3, 2), constrained_layout=True, sharey=True
    )
    # axes[0].set_xlabel("Displacement ($m$)")
    axes[0].set_ylabel("Velocity ($m/s$)")
    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[2].spines["left"].set_visible(False)
    axes[2].tick_params(axis="y", left=False)

    # No temporal forcing
    for k in (1, 3, 5):
        osc_period = 2 * np.pi * (m / k) ** 0.5
        runtime = Fraction(np.round(0.75 * osc_period, decimals=2)).limit_denominator(
            100
        )

        states = simulate_oscillator(x0, v0, k, m, dt, runtime)
        (lineplot,) = axes[0].plot(
            states[:, 1], states[:, 0], linewidth=1, label=f"$k_{k}$"
        )
        axes[0].annotate(
            "",
            xy=(states[-1, 1] + 1, states[-1, 0]),
            xytext=(states[-1, 1] - 0.5, states[-1, 0]),
            arrowprops={"arrowstyle": "->", "lw": 1, "color": lineplot.get_color()},
        )

    axes[0].legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8,
        handlelength=0.5,
        columnspacing=0.5,
        labelspacing=0.2,
        frameon=False,
    )

    k = 3
    forcings = [0.1, 0.9]
    colors = ["C3", "C4"]
    for forcing_multiplier, ax, color in zip(forcings, axes[1:], colors, strict=True):
        unforced_osc_period = 2 * np.pi * (m / k) ** 0.5
        runtime = Fraction(
            np.round(3 * unforced_osc_period, decimals=2)
        ).limit_denominator(100)

        forcing = forcing_multiplier * (m / k) ** 0.5
        states = simulate_oscillator(
            x0, v0, k, m, dt, runtime, forcing, forcing_amplitude=1.0
        )
        (lineplot,) = ax.plot(
            states[:, 1],
            states[:, 0],
            linewidth=1,
            color=color,
            label=f"$\\omega_F={forcing_multiplier:.2f}\\omega$",
        )
        ax.scatter(
            [states[0, 1]],
            [states[0, 0]],
            color="C2",
            edgecolor="black",
            linewidth=0.3,
            s=8,
            zorder=2,
        )
        ax.scatter(
            [states[-1, 1]],
            [states[-1, 0]],
            color="C0",
            edgecolor="black",
            linewidth=0.3,
            s=8,
            zorder=3,
        )

        ax.legend(
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            fontsize=8,
            handlelength=0.8,
            columnspacing=0.5,
            labelspacing=0.2,
            frameon=False,
        )

    fig.supxlabel("Displacement $(m)$")

    fig.savefig("results/figures/spring_1d_phaseplot.png", bbox_inches="tight")
    fig.savefig("results/figures/spring_1d_phaseplot.pdf", bbox_inches="tight")

"""Plot position and velocity of 1D spring under simple harmonic motion."""

from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from scicomp.oscillator.leapfrog import simulate_oscillator


def create_animation(
    x0: float,
    v0: float,
    k: float,
    m: float,
    dt: Fraction,
    runtime: int | Fraction,
    fps: int,
) -> FuncAnimation:
    """Create an animation of a 1D spring under simple harmonic motion.

    Args:
        x0: Initial position.
        v0: Initial velocity.
        k: Spring constant (N/m).
        m: Mass of object at the end of the spring (kg).
        dt: Duration of a discrete time-step.
        runtime: Total duration to simulate.
        fps: Animation framerate (frames/s).

    Returns:
        Matplotlib FuncAnimation object.
    """
    states = simulate_oscillator(x0, v0, k, m, dt, runtime)

    min_x = states[:, 1].min()
    max_x = states[:, 1].max()

    osc_period = 2 * np.pi * (m / k) ** 0.5

    # Prepare figure
    fig, ax = plt.subplots(figsize=(3, 1.5), constrained_layout=True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.set_xlabel("$x$")
    ax.set_title(f"Period of oscillation: {osc_period:.2f}s")

    # Add extra space to border so spring doesn't touch the sides
    extra_border = 0.1
    ax_min_x = min_x - extra_border * abs(min_x)
    ax_max_x = max_x + extra_border * abs(max_x)
    ax.set_xlim(ax_min_x, ax_max_x)

    # Plot spring as a line with a point mass
    x_states = states[:, 1]

    spring = ax.plot(
        [0, x_states[0]],
        [0, 0],
        color="grey",
        zorder=0,
    )[0]
    mass = ax.scatter(
        [x_states[0]],
        [[0]],
        color="red",
        zorder=2,
    )
    ax.scatter(
        [[0]],
        [[0]],
        color="black",
        s=5,
        zorder=1,
    )

    def update(frame):
        x_pos = x_states[frame].item()
        spring.set_xdata([0.0, x_pos])
        mass.set_offsets([[x_pos, 0.0]])
        return (spring, mass)

    return FuncAnimation(
        fig,
        update,
        frames=len(x_states),
        interval=1000 / fps,
        repeat=False,
        blit=True,
    )


def main():
    """Animate 1D spring."""
    m = 1
    k = 4
    x0 = 10
    v0 = 0
    dt = Fraction(1, 100)
    runtime = 5

    fig, ax = plt.subplots(figsize=(2.8, 2.8), constrained_layout=True)
    ax.set_xlabel("Position ($m$)")
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
        handlelength=1,
        labelspacing=0.2,
        frameon=False,
    )

    fig.savefig("results/figures/spring_1d_phaseplot.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

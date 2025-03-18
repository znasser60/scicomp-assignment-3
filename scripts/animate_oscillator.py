"""Create animation of 1D spring system under simple harmonic motion."""

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
    dt = Fraction(1, 200)
    runtime = Fraction(7, 2)

    ani = create_animation(x0, v0, k, m, dt, runtime, fps=dt.denominator)

    ani.save("results/animations/simple_spring.mp4", dpi=200)


if __name__ == "__main__":
    main()

"""Plot total energy over time for 1D spring simulation."""

from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

from scicomp.oscillator.energy import (
    calculate_elastic_potential_energy,
    calculate_kinetic_energy,
    calculate_spring_energy,
)
from scicomp.oscillator.leapfrog import simulate_oscillator


def main():
    """Plot the energy of a 1D spring over time."""
    m = 1
    k = 5
    x0 = 10
    v0 = 0
    dt = Fraction(1, 100)
    runtime = 5
    states = simulate_oscillator(x0, v0, k, m, dt, runtime)
    e_kinetic = calculate_kinetic_energy(states[:, 0], m)
    e_elastic = calculate_elastic_potential_energy(states[:, 1], k)
    e_total = calculate_spring_energy(states[:, 1], states[:, 0], k, m)

    fig, ax = plt.subplots(figsize=(2.8, 2.8), constrained_layout=True)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")

    time_points = np.linspace(0, runtime, states.shape[0])
    ax.plot(time_points, e_kinetic, label="$E_k$")
    ax.plot(time_points, e_elastic, label="$E_p$")
    ax.plot(time_points, e_total, label="$E$")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8,
        handlelength=1,
        labelspacing=0.2,
        frameon=False,
    )

    fig.savefig("results/figures/spring_1d_energy.pdf", bbox_inches="tight")
    fig.savefig("results/figures/spring_1d_energy.png", bbox_inches="tight")


if __name__ == "__main__":
    main()

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
    cycles = 2

    osc_period = 2 * np.pi * (m / k) ** 0.5
    runtime_frac = Fraction(
        np.round(osc_period * cycles, decimals=2)
    ).limit_denominator(100)
    runtime = float(runtime_frac)

    states = simulate_oscillator(x0, v0, k, m, dt, runtime_frac)
    e_kinetic = calculate_kinetic_energy(states[:, 0], m)
    e_elastic = calculate_elastic_potential_energy(states[:, 1], k)
    e_total = calculate_spring_energy(states[:, 1], states[:, 0], k, m)
    initial_energy = calculate_spring_energy(states[0, 1], states[0, 0], k, m)

    fig, ax = plt.subplots(figsize=(2.8, 2.8), constrained_layout=True)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")

    time_points = np.linspace(0, runtime, states.shape[0])
    ax.plot(time_points, e_kinetic, linewidth=1, label="$E_k$")
    ax.plot(time_points, e_elastic, linewidth=1, label="$E_p$")
    ax.plot(time_points, e_total, linewidth=1, label="$E$")
    ax.axhline(
        y=initial_energy,
        linewidth=0.5,
        linestyle="dashed",
        color="black",
        label="$E_0$",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        ncol=4,
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

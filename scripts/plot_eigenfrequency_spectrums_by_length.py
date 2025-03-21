"""Plot spectrum of eigenfrequencies for varying N."""

import argparse
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scicomp.domains import Circle


def main(n_at_unit_length: int, quality_label: str):
    """Plot spectrum of eigenfrequencies for varying length."""
    h = Fraction(1, n_at_unit_length)
    k = n_at_unit_length - 1

    lengths = [Fraction(length, 10) for length in np.arange(10, 51, 5)]

    data = {
        "length": [],
        "lambda": [],
    }
    for length in lengths:
        domain = Circle(length)
        n = (length / h).numerator
        eigenfrequencies, _ = domain.solve_eigenproblem(
            k, n, use_sparse=True, shift_invert=True
        )
        data["length"].extend(np.repeat(float(length), len(eigenfrequencies)).tolist())
        data["lambda"].extend(eigenfrequencies.tolist())

    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
    sns.boxplot(
        x=data["length"],
        y=data["lambda"],
        fill=False,
        color="grey",
        width=0.2,
        native_scale=True,
        ax=ax,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(0, None)

    ax.set_xlabel("Object length ($m$)")
    ax.set_ylabel("$\\lambda$ (hz)")

    fig.savefig(
        f"results/figures/eigenfrequency_spectrum_by_length_{quality_label}_quality.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-at-unit-length", type=int)
    parser.add_argument("--quality-label", type=str)
    args = parser.parse_args()
    main(args.n_at_unit_length, args.quality_label)

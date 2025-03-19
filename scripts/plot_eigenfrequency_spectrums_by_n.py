"""Plot spectrum of eigenfrequencies for varying N."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from scicomp.domains import Circle


def main(min_n: int, max_n: int):
    """Plot spectrum of eigenfrequencies for varying N."""
    length = 1

    k1 = min_n - 1

    data = {
        "n": [],
        "lambda": [],
        "first_k1": [],
    }
    domain = Circle(length)
    ns = [min_n, max_n]
    for n in ns:
        k = n - 1
        eigenfrequencies, _ = domain.solve_eigenproblem(
            k, n, use_sparse=True, shift_invert=True
        )
        data["n"].extend(np.repeat(n, len(eigenfrequencies)).tolist())
        data["lambda"].extend(eigenfrequencies.tolist())
        first_k_mask = np.arange(len(eigenfrequencies)) >= k1
        data["first_k1"].extend(first_k_mask.tolist())

    fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)
    sns.swarmplot(
        x=data["n"],
        y=data["lambda"],
        hue=data["first_k1"],
        s=2.5,
        legend=False,
        ax=ax,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(0, None)

    ax.set_xlabel("# Discretisation intervals")
    ax.set_ylabel("$\\lambda_i$ (hz)")

    handles = [
        Line2D([], [], marker=".", color="tab:blue", linestyle="None"),
        Line2D([], [], marker=".", color="tab:orange", linestyle="None"),
    ]
    ax.legend(
        handles=handles,
        labels=[f"$i < {min_n}$", f"$i \\geq {min_n}$"],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8,
        labelspacing=0.2,
        frameon=False,
    )

    fig.savefig(
        "results/figures/eigenfrequency_spectrum_by_n.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int)
    parser.add_argument("--max-n", type=int)
    args = parser.parse_args()
    main(args.min_n, args.max_n)

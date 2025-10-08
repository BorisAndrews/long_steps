"""Plot the family of polynomials p_k(x) defined by

    p_0(x) = 1,
    p_k(x) = p_{k-1}(x) * (1 - x / sin(phi * pi * k)^2)

where phi is the golden ratio. The script evaluates the products on a grid
and plots p_0 .. p_n for a user-specified n (default 20).

Example::

    python eigenvalue_maps.py --n 20 --outfile eigen_maps_n20.png

This will save the figure to the given file. Use --show to pop up an
interactive window (if available).
"""
from __future__ import annotations

import argparse
import math
from typing import Iterable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def sin_sq(phi: float, k: int) -> float:
    """
    Return sin(phi * pi * k) squared.
    """
    return math.sin(phi * math.pi * k) ** 2


def evaluate_family(x: np.ndarray, n: int, phi: float) -> list[np.ndarray]:
    """
    Evaluate p_0 .. p_n on the grid x.

    Returns a list of arrays [p_0(x), p_1(x), ..., p_n(x)].
    """
    p_list: list[np.ndarray] = []
    vals = np.ones_like(x, dtype=float)  # p_0
    p_list.append(vals.copy())

    for k in range(1, n + 1):
        s2 = sin_sq(phi, k)
        # factor = (1 - x / s2)
        vals = vals * (1.0 - x / (0.0 + 1.0*s2))
        p_list.append(vals.copy())

    return p_list


def roots_list(n: int, phi: float) -> list[float]:
    """
    Return the list of roots x = sin(phi*pi*k)^2 for k=1..n.
    """
    return [sin_sq(phi, k) for k in range(1, n + 1)]


def plot_family(x: np.ndarray, p_list: Iterable[np.ndarray], roots: Iterable[float], outfile: str | None = None, show_roots: bool = True) -> None:
    """
    Plot the family of polynomials and optionally mark roots.

    The newest polynomial (highest index) is drawn with a thicker line.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    p_arrs = list(p_list)
    n = len(p_arrs) - 1

    # Use non-deprecated colormap access
    cmap = matplotlib.colormaps.get_cmap("viridis")
    for k, arr in enumerate(p_arrs):
        color = cmap(k / max(1, n))
        lw = 2.0 if k == n else 1.0
        alpha = 0.95 if k == n else 0.6
        ax.plot(x, arr, color=color, linewidth=lw, alpha=alpha)

    ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)

    if show_roots:
        # mark roots on x-axis
        roots = np.array(list(roots))
        # only show roots inside the plotted x-range
        x_min, x_max = ax.get_xlim()
        mask = (roots >= x_min) & (roots <= x_max)
        visible_roots = roots[mask]
        if visible_roots.size:
            y_min, y_max = ax.get_ylim()
            # place small markers at y=0
            ax.scatter(visible_roots, np.zeros_like(visible_roots), marker="x", color="red", zorder=5, label="roots")

    ax.set_xlabel("x")
    ax.set_ylabel("p_k(x)")
    ax.set_title("Family of p_k(x) for k=0..{}".format(n))
    # annotate last curve
    ax.text(0.02, 0.95, f"highest: p_{n}", transform=ax.transAxes, ha="left", va="top")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200)
        print(f"Saved plot to: {outfile}")
    else:
        print("No outfile specified; not saving.")


def compute_maxima(x: np.ndarray, p_list: Iterable[np.ndarray]) -> list[float]:
    """
    Compute M_k = max_x ( |x * p_k(x)^2| ) for each k using the provided grid.
    """
    x_arr = np.asarray(x, dtype=float)
    mask = x_arr >= 0.0

    maxima: list[float] = []
    for p in p_list:
        vals = np.abs(x_arr[mask] * (p[mask] ** 2))
        # handle empty mask (e.g., all x negative)
        m = float(np.max(vals)) if vals.size else float("nan")
        maxima.append(m)
    return maxima


def plot_maxima(maxima: Iterable[float], outfile: str | None = None, log: bool = False) -> None:
    """
    Plot the sequence 1 / M_k vs k, where M_k = max_x |x * p_k(x)^2|.
    """
    seq = np.asarray(list(maxima), dtype=float)
    k = np.arange(seq.size)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k, seq, marker="o", linewidth=1.5)
    ax.plot(k[1:], seq[122]*122**2 * 1/k[1:]**2, linestyle="--", linewidth=1.5)
    ax.set_xlabel("k")
    ax.set_ylabel("1 / max over x of |x * p_k(x)^2|")
    ax.set_title("Inverse maxima of |x * p_k(x)^2| vs k")
    # force log scale if requested
    if log:
        ax.set_yscale("log")
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200)
        print(f"Saved plot to: {outfile}")
    else:
        print("No outfile specified; not saving.")


def poly_coeffs_for_pk(k: int, phi: float) -> np.ndarray:
    """
    Return ascending-order polynomial coefficients for p_k(x) = prod_{j=1..k} (1 - a_j x), a_j = 1/sin(phi*pi*j)^2.

    Ascending order means coeffs[i] is the coefficient of x**i.
    """
    coeffs = np.array([1.0], dtype=float)  # p_0(x) = 1
    for j in range(1, k + 1):
        s2 = sin_sq(phi, j)
        a = 1.0 / (0.0 + 1.0*s2)
        # multiply by (1 - a x): convolution with [1, -a]
        coeffs = np.convolve(coeffs, np.array([1.0, -a], dtype=float))
    return coeffs


def compute_maxima_polyroots(n: int, phi: float) -> list[float]:
    """
    Compute M_k = max_{x in [0,1]} | x * p_k(x)^2 | using derivative roots.

    Builds polynomial coefficients explicitly, finds real critical points of q_k(x)=x*p_k(x)^2 in the domain,
    and evaluates to get maxima. Returns list for k=0..n.
    """
    maxima: list[float] = []
    # k = 0 case: q_0(x) = x
    # Max over [0, 1]
    maxima.append(1)

    for k in range(1, n + 1):
        pk = poly_coeffs_for_pk(k, phi)  # ascending
        pk2 = np.convolve(pk, pk)  # ascending
        # q_k(x) = x * p_k(x)^2 => ascending coeffs shift by 1
        qk = np.concatenate(([0.0], pk2))
        # derivative in ascending order: c'[i-1] = i*c[i]
        deriv = np.array([i * qk[i] for i in range(1, qk.size)], dtype=float)
        # roots via numpy (expects descending order)
        deriv_desc = deriv[::-1]
        try:
            roots = np.roots(deriv_desc)
        except Exception:
            roots = np.array([], dtype=complex)

        # collect candidate x's: endpoints + real roots in domain
        candidates = [0, 1]
        for r in roots:
            if np.isfinite(r) and abs(r.imag) < 1e-9:
                xr = float(r.real)
                if - 1e-12 <= xr <= 1 + 1e-12:
                    candidates.append(xr)

        # evaluate |q_k(x)| at candidates and take max
        # use numpy.polyval which expects descending order
        qk_desc = qk[::-1]
        vals = []
        for xc in candidates:
            vals.append(abs(np.polyval(qk_desc, xc)))
        mk = float(max(vals)) if vals else float("nan")
        maxima.append(mk)

    return maxima


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eigenvalue-style polynomial family p_k(x) for k=0..n.")
    parser.add_argument("--n", type=int, default=610, help="highest k to compute (default 20)")
    parser.add_argument("--points", type=int, default=1000000, help="number of x points to evaluate (default 2000)")
    parser.add_argument("--mode", type=str, choices=["family", "maxima"], default="maxima", help="plot mode: 'family' to plot p_k(x) curves, 'maxima' to plot 1/max |x*p_k(x)^2| vs k (default maxima)")
    parser.add_argument("--log", action="store_true", help="use log scales for axes in maxima mode (recommended)")
    parser.add_argument("--maxima-method", type=str, choices=["grid", "polyroots"], default="grid", help="how to compute maxima in maxima mode: 'grid' uses sampled x, 'polyroots' uses polynomial derivative roots (default)")
    parser.add_argument("--outfile", type=str, default="eigen_maps_n{n}.png", help="output filename (default 'eigen_maps_n{n}.png')")
    parser.add_argument("--show", action="store_true", help="display the interactive plot window after saving")
    parser.add_argument("--no-roots", dest="show_roots", action="store_false", help="(family mode) do not mark the roots on the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    n = args.n

    # format default outfile if user didn't supply an explicit string
    outfile = args.outfile
    if outfile and "{n}" in outfile:
        outfile = outfile.format(n=n)

    x = np.linspace(0, 1, args.points, dtype=float)
    p_list = evaluate_family(x, n=n, phi=phi)

    if args.mode == "family":
        roots = roots_list(n=n, phi=phi)
        plot_family(x, p_list, roots, outfile=outfile, show_roots=args.show_roots)
    else:
        # maxima mode
        if args.maxima_method == "polyroots":
            maxima = compute_maxima_polyroots(n=n, phi=phi)
        else:
            maxima = compute_maxima(x, p_list)
        # plot inverse maxima with log scale (default on if --log set)
        plot_maxima(maxima, outfile=outfile, log=args.log or True)

    if args.show:
        try:
            plt.show()
        except Exception as e:
            print("Could not show interactive window:", e)


if __name__ == "__main__":
    main()

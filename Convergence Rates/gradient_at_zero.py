"""
Plot positive integers n against the running average

		A(n) = (1/n) * sum_{i=1..n} 1/sin(pi * phi * i)^2,

where phi is the golden ratio.

Usage (PowerShell):
	python convergence_rates.py --N 1000
	python convergence_rates.py --N 5000 --save "golden_sine_inverse_squared_avg.png"

Notes:
- Values can be extremely large when sin(pi*phi*i) is close to 0. Consider using --log to view in log scale.
- Computation is efficient using vectorized numpy and cumulative sums (no recomputation per n).
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def compute_sequence(N: int) -> tuple[np.ndarray, np.ndarray]:
	"""Compute n and A(n) = (1/n) * sum_{i=1..n} 1/sin(pi * phi * i)^2 for n = 1..N.

	Parameters
	----------
	N: int
		Maximum positive integer to include (must be >= 1).

	Returns
	-------
	n_vals: np.ndarray
		Integers 1..N (shape: (N,))
	a_vals: np.ndarray
		Running averages A(n) (shape: (N,))
	"""
	if N < 1:
		raise ValueError("N must be a positive integer (>= 1)")

	n_vals = np.arange(1, N + 1, dtype=np.int64)
	phi = (1.0 + math.sqrt(5.0)) / 2.0

	# Compute b_i = 1/sin(pi * phi * i)^2 for i = 1..N
	s = np.sin(np.pi * phi * n_vals)

	# Avoid exact zeros (shouldn't occur for irrational multiples, but pay attention to underflow)
	eps = np.finfo(np.float64).tiny
	s = np.where(np.abs(s) < eps, np.sign(s) * eps, s)

	b_vals = 1.0 / (s * s)

	# Efficient running average: A(n) = (1/n) * cumsum(b_vals)[n-1]
	cums = np.cumsum(b_vals, dtype=np.float64)
	a_vals = cums / n_vals
	return n_vals, a_vals


def plot_sequence(n_vals: np.ndarray, y_vals: np.ndarray, *, log: bool = False, save: Optional[str] = None) -> None:
	"""Plot n vs A(n) running average.

	Parameters
	----------
	n_vals: np.ndarray
		1..N
	y_vals: np.ndarray
		A(n) = (1/n) * sum_{i=1..n} 1/sin(pi * phi * i)^2
	log: bool
		If True, use log scale on y-axis.
	save: Optional[str]
		If provided, path to save the figure.
	"""
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(n_vals, y_vals, lw=1.0)
	ax.set_xlabel("n")
	ax.set_ylabel(r"$\frac{1}{n}\sum_{i=1}^{n} \frac{1}{\sin^{2}(\pi\,\varphi\,i)}$")
	ax.set_title("Running average of 1/sin^2(pi * phi * i)")
	ax.grid(True, which="both", ls=":", alpha=0.6)

	if log:
		ax.set_yscale("log")

	if save:
		fig.tight_layout()
		fig.savefig(save, dpi=150)

	plt.show()


def main(argv: Optional[list[str]] = None) -> None:
	parser = argparse.ArgumentParser(description=(
		"Plot n vs 1/sin(pi*phi*n)^2 for the golden ratio phi."
	))
	parser.add_argument("--N", type=int, default=1000, help="Max n (positive integer, default 1000)")
	parser.add_argument("--log", action="store_true", help="Use log scale for y-axis")
	parser.add_argument("--save", type=str, default=None, help="Optional path to save the plot image")

	args = parser.parse_args(argv)

	n_vals, y_vals = compute_sequence(args.N)
	plot_sequence(n_vals, y_vals, log=args.log, save=args.save)


if __name__ == "__main__":
	main()


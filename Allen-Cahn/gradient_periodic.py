#!/usr/bin/env python3

"""
Gradient descent for stationary Allen–Cahn with H1-seminorm descent direction and
periodic, user-specified step sizes (no line search).

Energy:
  E(u) = ∫ [ 1/4 (u^2 - 1)^2 + (1/(2γ)) |∇u|^2 ] dx

Setup:
  - Dirichlet BCs: u = 0 on ∂Ω (unit square).
  - Mass constraint enforced in the gradient direction via a mixed Riesz map so ∫ g dx = 0.
  - User passes a sequence of tau values; iterations cycle through them.
  - Writes u every N iterations to a PVD time series.
"""

from firedrake import (
    UnitSquareMesh,
    FunctionSpace,
    Function,
    TestFunctions,
    TrialFunctions,
    LinearVariationalProblem,
    LinearVariationalSolver,
    Constant,
    assemble,
    grad,
    dot,
    inner,
    dx,
    VTKFile,
    SpatialCoordinate,
    sin,
    DirichletBC,
    pi,
    RED,
    GREEN,
)
import argparse
import numpy.random as rand


# Editable dictionary of named tau sequences.
# Add or modify entries here; values are lists of step sizes cycled in order.
TAU_SEQUENCES = {
    "1": [1.0,],
    "3": [1.5, 4.9, 1.5],
    "7": [1.5, 2.2, 1.5, 12.0, 1.5, 2.2, 1.5],
    "15": [1.4, 2.0, 1.4, 4.5, 1.4, 2.0, 1.4, 29.7, 1.4, 2.0, 1.4, 4.5, 1.4, 2.0, 1.4],
    "31": [1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 8.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 72.3, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 8.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4],
    "63": [1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 14.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 164.0, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 14.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4],
    "127": [1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 23.5, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 370.0, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 23.5, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 12.6, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4, 7.2, 1.4, 2.0, 1.4, 3.9, 1.4, 2.0, 1.4],
}


def energy_form(u, gamma):
    return (0.25 * (u**2 - 1.0)**2 + 0.5 / Constant(gamma) * dot(grad(u), grad(u))) * dx


def gradient_direction(u, gamma):
    """
    Compute H1-seminorm gradient direction g of E at current u with g=0 on ∂Ω and ∫ g dx = 0:
        Find (g, α) in V×R such that
            (∇g, ∇v) + α ∫ v dx + q ∫ g dx = dE/du[v]
    Returns g (mean-zero step direction).
    """
    V = u.function_space()
    R = FunctionSpace(V.mesh(), "R", 0)
    W = V * R

    (g, alpha) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    gamma_c = Constant(gamma)
    L = ((u**2 - 1.0) * u) * v * dx + (1.0 / gamma_c) * dot(grad(u), grad(v)) * dx
    a = inner(grad(g), grad(v)) * dx + alpha * v * dx + q * g * dx

    w = Function(W)
    bc = DirichletBC(W.sub(0), 0.0, "on_boundary")
    problem = LinearVariationalProblem(a, L, w, bcs=bc)
    solver = LinearVariationalSolver(problem)
    solver.solve()
    g_sol, _alpha = w.subfunctions
    return g_sol


def parse_taus(taus_str: str):
    if not taus_str or not taus_str.strip():
        raise ValueError("--taus must be a non-empty comma-separated list of step sizes")
    parts = [p.strip() for p in taus_str.split(",")]
    taus = []
    for p in parts:
        if not p:
            continue
        taus.append(float(eval(p, {}, {})) if any(ch.isalpha() for ch in p) else float(p))
    if not taus:
        raise ValueError("Parsed empty tau list from --taus")
    return taus


def gradient_descent_periodic(nx=128, degree=1, gamma=2**9, taus=None,
                              max_iters=2**10,
                              ic_max=6, ic_amp=2**-8, seed=None,
                              output="gradient_periodic.pvd", save_every=1):
    if not taus:
        raise ValueError("Provide at least one step size via --taus, e.g. --taus 1e-2,5e-3,1e-2")

    # Mesh and space
    mesh = UnitSquareMesh(nx, nx, quadrilateral=True)
    V = FunctionSpace(mesh, "Q", degree)
    print(RED % f"Degrees of freedom: {V.dim()}")
    bc = DirichletBC(V, 0.0, "on_boundary")

    # Initial guess: sum over all sine modes sin(Aπx)sin(Bπy) up to max A,B (Dirichlet-compatible)
    x, y = SpatialCoordinate(mesh)
    u = Function(V, name="u")
    rng = rand.RandomState(seed) if seed is not None else rand
    base = 0
    for Ax in range(1, int(ic_max) + 1):
        for By in range(1, int(ic_max) + 1):
            coeff = float(rng.standard_normal())
            base = base + coeff * sin(Ax * pi * x) * sin(By * pi * y) / (Ax**2 + By**2)**1.5
    u.interpolate(ic_amp * base)
    bc.apply(u)

    # Energy and output
    E = lambda uu: assemble(energy_form(uu, gamma))
    writer = VTKFile("output/" + output)
    writer.write(u, time=float(0))

    E_curr = E(u)
    mean_u = assemble(u * dx)
    print(GREEN % f"iter 0: E={E_curr:.12e}, mean(u)={mean_u:.12e}")

    # Descent loop with periodic taus
    Lip = 2.0**2/(2 * pi) + 1/gamma
    nT = len(taus)
    for k in range(1, max_iters + 1):
        tau_k = float(taus[(k - 1) % nT])

        # Gradient direction (zero boundary, zero mean)
        g = gradient_direction(u, gamma)
        g_norm2 = assemble(dot(grad(g), grad(g)) * dx)
        if g_norm2 <= 1e-30:
            print(GREEN % f"iter {k}: gradient ~ 0, stopping.")
            break

        # Fixed step update
        u.assign(u - tau_k / Lip * g)
        bc.apply(u)
        E_curr = E(u)
        mean_u = assemble(u * dx)
        print(GREEN % f"iter {k}: E={E_curr:.12e}, mean(u)={mean_u:.12e}, tau={tau_k:.3e}, |g|_H1^2={g_norm2:.3e}")

        if save_every and (k % save_every == 0):
            writer.write(u, time=float(k))

    # Final save if not already saved at last iteration
    if (max_iters % save_every) != 0:
        writer.write(u, time=float(k))


def parse_args():
    p = argparse.ArgumentParser(description="H1-seminorm gradient descent with periodic step sizes for Allen–Cahn (Dirichlet BCs, mass-constrained gradient)")
    p.add_argument("--nx", type=int, default=2**4, help="Number of elements (in each direction)")
    p.add_argument("--degree", type=int, default=2**2, help="Polynomial degree for u")
    p.add_argument("--gamma", type=float, default=2**9, help="Gradient penalty coefficient (interface width^2)")
    # Either explicit taus, a named dictionary sequence, or a preset can be provided
    p.add_argument("--taus", type=str, default=None, help="Comma-separated list of step sizes, e.g. '1e-2,5e-3,1e-2'")
    p.add_argument("--cycle-length", type=str, default=None, help="Length of tau cycle in TAU_SEQUENCES dict")
    p.add_argument("--max-iters", type=int, default=2**10, help="Maximum iterations")
    p.add_argument("--ic-max", type=int, default=6, help="Max frequency index for A and B in sin(Aπx) and sin(Bπy)")
    p.add_argument("--ic-amp", type=float, default=2**-8, help="Overall amplitude scaling for IC")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for IC")
    p.add_argument("--output", type=str, default="gradient_periodic.pvd", help="Output VTK/PVD filename for u trajectory")
    p.add_argument("--save-every", type=int, default=1, help="Save every N iterations")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.taus:
        taus = parse_taus(args.taus)
    elif args.cycle_length:
        key = args.cycle_length.strip()
        if key not in TAU_SEQUENCES:
            available = ", ".join(sorted(TAU_SEQUENCES.keys()))
            raise SystemExit(f"Unknown --cycle-length '{key}'. Available: {available}")
        taus = TAU_SEQUENCES[key]
    else:
        raise SystemExit("Please provide either --taus or --cycle-length.")
    gradient_descent_periodic(
        nx=args.nx,
        degree=args.degree,
        gamma=args.gamma,
        taus=taus,
        max_iters=args.max_iters,
        ic_max=args.ic_max,
        ic_amp=args.ic_amp,
        seed=args.seed,
        output=args.output,
        save_every=args.save_every,
    )

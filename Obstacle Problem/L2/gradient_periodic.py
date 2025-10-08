#!/usr/bin/env python3

"""
Gradient descent for constrained optimization with H1-seminorm descent direction and
periodic, user-specified step sizes (no line search).

Energy:
  E(u) = (1/2) ∫ max(f - u, 0)^2 dx + (1/(2γ)) ∫ |∇u|^2 dx

Setup:
  - Dirichlet BCs: u = 0 on ∂Ω (unit square).
  - User passes a sequence of tau values; iterations cycle through them.
  - Writes u every N iterations to a PVD time series.
"""

from firedrake import (
    UnitSquareMesh,
    FunctionSpace,
    Function,
    TestFunction,
    TrialFunction,
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
    cos,
    DirichletBC,
    pi,
    RED,
    GREEN,
    conditional,
    gt,
    exp,
    sqrt,
    cosh,
    ln,
)
import argparse
import numpy.random as rand
import csv
import os


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


def energy_form(u, f, gamma):
    """Constrained optimization energy: (1/2) max(f - u, 0)^2 + (1/(2γ)) |∇u|^2"""
    constraint_violation = conditional(gt(f - u, 0), f - u, 0)
    return 0.5 * constraint_violation**2 + 0.5 / Constant(gamma) * dot(grad(u), grad(u))


def gradient_direction(u, f, gamma, f_center=None, f_width=None, circular_inner_product=False):
    """
    Compute gradient direction g of E at current u with g=0 on ∂Ω:
        Find g in V such that
            (g, v)_* = dE/du[v]
    where (·,·)_* is either the H1-seminorm inner product or the circular inner product.
    Returns g (step direction).
    """
    V = u.function_space()

    g = TrialFunction(V)
    v = TestFunction(V)

    gamma_c = Constant(gamma)
    
    # First variation for constrained energy
    # d/du[(1/2) max(f - u, 0)^2] = -max(f - u, 0) * H(f - u) where H is Heaviside
    constraint_viol = conditional(gt(f - u, 0), f - u, 0)
    heaviside = conditional(gt(f - u, 0), 1.0, 0.0)
    L = -constraint_viol * heaviside * v * dx + (1.0 / gamma_c) * dot(grad(u), grad(v)) * dx
    
    # Choose inner product for gradient computation
    if circular_inner_product and f_center is not None and f_width is not None:
        # Use L2 inner product on circular region + H1-seminorm
        x, y = SpatialCoordinate(V.mesh())
        cx, cy = f_center
        radius = f_width
        distance_sq = (x - cx)**2 + (y - cy)**2
        circular_region = conditional(distance_sq <= radius**2, 1.0, 0.0)
        
        a = circular_region * g * v * dx + (1.0 / gamma_c) * inner(grad(g), grad(v)) * dx
    else:
        # Standard H1 inner product
        a = (g * v + (1.0 / gamma_c) * inner(grad(g), grad(v))) * dx

    g_sol = Function(V)
    bc = DirichletBC(V, 0.0, "on_boundary")
    problem = LinearVariationalProblem(a, L, g_sol, bcs=bc)
    solver = LinearVariationalSolver(problem)
    solver.solve()
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


def gradient_descent_periodic(nx=128, degree=1, gamma=2**16, taus=None, chebyshev_taus_rate=None, chebyshev_taus_prob=None,
                                          f_type="gaussian", f_center=(0.5, 0.5), f_width=2**-3, f_amp=1.0, f_shift=-2**-4,
                                          max_iters=2**10,
                                          ic_max=6, ic_amp=0, seed=None,
                                          output="gradient_periodic.pvd", save_every=1, energy_log=None,
                                          circular_inner_product=False):
    if not taus and not chebyshev_taus_rate:
        raise ValueError("Provide at least one step size via --taus, e.g. --taus 1e-2,5e-3,1e-2, or rate for Chebyshev tau cycles via --chebyshev-taus-rate, e.g. --chebyshev-taus-rate 0.9")

    # Mesh and space
    mesh = UnitSquareMesh(nx, nx, quadrilateral=True)
    V = FunctionSpace(mesh, "Q", degree)
    print(RED % f"Degrees of freedom: {V.dim()}")
    bc = DirichletBC(V, 0.0, "on_boundary")

    # Define constraint function f(x,y)
    x, y = SpatialCoordinate(mesh)
    cx, cy = f_center
    
    if f_type == "gaussian":
        sigma = f_width
        f = f_amp * exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2)) + f_shift
    elif f_type == "circular":
        radius = f_width
        distance_sq = (x - cx)**2 + (y - cy)**2
        f = f_amp * conditional(distance_sq <= radius**2, 1.0, 0.0) + f_shift
    else:
        raise ValueError(f"Unknown f_type: {f_type}. Must be 'gaussian' or 'circular'.")

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
    E = lambda uu: assemble(energy_form(uu, f, gamma) * dx)
    writer = VTKFile("output/" + output)
    writer.write(u, time=float(0))

    # Initialize energy logging
    energy_data = []
    if energy_log:
        os.makedirs("output", exist_ok=True)

    E_curr = E(u)
    constraint_viol = conditional(gt(f - u, 0), f - u, 0)
    total_violation = assemble(constraint_viol * dx)
    print(GREEN % f"iter 0: E={E_curr:.12e}, violation={total_violation:.12e}")
    
    # Record initial energy
    energy_data.append({'iteration': 0, 'energy': float(E_curr), 'violation': float(total_violation), 'tau': 0.0, 'gradient_norm': 0.0})

    # Descent loop with periodic taus
    Lip = Constant(1)  # Exact Lipschitz value

    if chebyshev_taus_rate:
        phi = (1 + sqrt(5))/2
        chebyshev_taus_scale = 2/(1 + cosh(ln(chebyshev_taus_rate)))
    
    if taus:
        nT = len(taus)
    
    if chebyshev_taus_prob:
        k_chebyshev = 1

    for k in range(1, max_iters + 1):
        if chebyshev_taus_prob:
            if rng.uniform() > chebyshev_taus_prob:
                tau_k = float(taus[(k - k_chebyshev) % nT])
            else:
                tau_k = 1/(1 - chebyshev_taus_scale * cos(pi*phi*k_chebyshev)**2)
                k_chebyshev += 1
        elif taus and not(chebyshev_taus_rate):
            tau_k = float(taus[(k - 1) % nT])
        else:
            tau_k = 1/(1 - chebyshev_taus_scale * cos(pi*phi*k)**2)
        
        # Gradient direction (zero boundary)
        g = gradient_direction(u, f, gamma, f_center, f_width, circular_inner_product)
        g_norm2 = assemble(dot(grad(g), grad(g)) * dx)
        if g_norm2 <= 1e-30:
            print(GREEN % f"iter {k}: gradient ~ 0, stopping.")
            break

        # Fixed step update
        u.assign(u - tau_k / Lip * g)
        bc.apply(u)
        E_curr = E(u)
        constraint_viol = conditional(gt(f - u, 0), f - u, 0)
        total_violation = assemble(constraint_viol * dx)
        print(GREEN % f"iter {k}: E={E_curr:.12e}, violation={total_violation:.12e}, tau={tau_k:.3e}, |g|_H1^2={g_norm2:.3e}")
        
        # Record energy data
        energy_data.append({
            'iteration': k, 
            'energy': float(E_curr), 
            'violation': float(total_violation), 
            'tau': float(tau_k), 
            'gradient_norm': float(g_norm2)
        })

        if save_every and (k % save_every == 0):
            writer.write(u, time=float(k))

    # Final save if not already saved at last iteration
    if (max_iters % save_every) != 0:
        writer.write(u, time=float(k))
    
    # Save energy data to CSV
    if energy_log:
        csv_file = f"output/{energy_log}"
        with open(csv_file, 'w', newline='') as f:
            writer_csv = csv.DictWriter(f, fieldnames=['iteration', 'energy', 'violation', 'tau', 'gradient_norm'])
            writer_csv.writeheader()
            writer_csv.writerows(energy_data)
        print(GREEN % f"Energy data saved to {csv_file}")


def parse_args():
    p = argparse.ArgumentParser(description="H1-seminorm gradient descent with periodic step sizes for Gaussian-constrained optimization (Dirichlet BCs)")
    p.add_argument("--nx", type=int, default=2**4, help="Number of elements (in each direction)")
    p.add_argument("--degree", type=int, default=2**2, help="Polynomial degree for u")
    p.add_argument("--gamma", type=float, default=2**16, help="Gradient penalty coefficient (interface width^2)")
    p.add_argument("--f-type", type=str, default="circular", choices=["gaussian", "circular"], help="Type of constraint function: gaussian or circular")
    p.add_argument("--f-center", type=float, nargs=2, default=[0.4, 0.7], help="Center (x,y) for constraint")
    p.add_argument("--f-width", type=float, default=2**-3, help="Width parameter (sigma for gaussian, radius for circular)")
    p.add_argument("--f-amp", type=float, default=1.0, help="Amplitude of gaussian constraint")
    p.add_argument("--f-shift", type=float, default=-2**-4, help="Vertical shift for gaussian constraint (negative shifts below plane)")
    # Either explicit taus, a named dictionary sequence, or a preset can be provided
    p.add_argument("--taus", type=str, default=None, help="Comma-separated list of step sizes, e.g. '1e-2,5e-3,1e-2'")
    p.add_argument("--taus-cycle", type=str, default=None, help="Length of tau cycle in TAU_SEQUENCES dict")
    p.add_argument("--chebyshev-taus-rate", type=float, default=None, help="Selected if using Chebyshev tau cycles, value (in [0,1]) indicates target rate for exponential decay of low-frequency modes")
    p.add_argument("--chebyshev-taus-prob", type=float, default=None, help="Probability of using taking tau from the Chebyshev tau cycle")
    p.add_argument("--max-iters", type=int, default=2**10, help="Maximum iterations")
    p.add_argument("--ic-max", type=int, default=6, help="Max frequency index for A and B in sin(Aπx) and sin(Bπy)")
    p.add_argument("--ic-amp", type=float, default=0, help="Overall amplitude scaling for IC")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for IC")
    p.add_argument("--output", type=str, default="gradient_periodic.pvd", help="Output VTK/PVD filename for u trajectory")
    p.add_argument("--save-every", type=int, default=1, help="Save every N iterations")
    p.add_argument("--energy-log", type=str, default=None, help="CSV filename to save energy history (e.g., 'energies_cycle3.csv')")
    p.add_argument("--circular-inner-product", action="store_true", help="Use circular inner product (L2 on circle + H1-seminorm) instead of H1-seminorm")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.taus and args.taus_cycle:
        raise SystemExit(f"The arguments --taus and --taus-cycle cannot both be specified.")
    elif not(args.taus or args.taus_cycle) and not(args.chebyshev_taus_rate):
        raise SystemExit(f"At least one of --taus, --taus-cycle, or --chebyshev-taus-rate must be specified.")
    elif (args.taus or args.taus_cycle) and args.chebyshev_taus_rate and not(args.chebyshev_taus_prob):
        raise SystemExit(f"If --taus or --taus-cycle and --chebyshev-taus-rate are specified, --chebyshev-taus-prob must be too.")
    elif (args.taus or args.taus_cycle) and not(args.chebyshev_taus_rate) and args.chebyshev_taus_prob:
        raise SystemExit(f"If --taus or --taus-cycle and --chebyshev-taus-prob are specified, --chebyshev-taus-rate must be too.")
    elif not(args.taus or args.taus_cycle) and args.chebyshev_taus_rate and args.chebyshev_taus_prob:
        raise SystemExit(f"If --chebyshev-taus-rate and --chebyshev-taus-prob are specified, either --taus or --taus-cycle must be too.")
    
    if args.taus:
        taus = parse_taus(args.taus)
    elif args.taus_cycle:
        key = args.taus_cycle.strip()
        if key not in TAU_SEQUENCES:
            available = ", ".join(sorted(TAU_SEQUENCES.keys()))
            raise SystemExit(f"Unknown --taus-cycle '{key}'. Available: {available}")
        taus = TAU_SEQUENCES[key]
    else:
        taus = None

    gradient_descent_periodic(
        nx=args.nx,
        degree=args.degree,
        gamma=args.gamma,
        taus=taus,
        chebyshev_taus_rate=args.chebyshev_taus_rate,
        chebyshev_taus_prob=args.chebyshev_taus_prob,
        f_type=args.f_type,
        f_center=tuple(args.f_center),
        f_width=args.f_width,
        f_amp=args.f_amp,
        f_shift=args.f_shift,
        max_iters=args.max_iters,
        ic_max=args.ic_max,
        ic_amp=args.ic_amp,
        seed=args.seed,
        output=args.output,
        save_every=args.save_every,
        energy_log=args.energy_log,
        circular_inner_product=args.circular_inner_product,
    )
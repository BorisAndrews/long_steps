#!/usr/bin/env python3

"""
Gradient descent for the constrained optimization problem under the H1 seminorm with
homogeneous Dirichlet boundary conditions (u = 0 on ∂Ω).

We minimize
    E(u) = (1/2) ∫ max(f - u, 0)^2 dx + (1/(2γ)) ∫ |∇u|^2 dx

We compute the H1-seminorm gradient g at u by solving the Riesz representation:
    find g in H1_0 s.t. (∇g, ∇v) = dE/du[v]
We then perform a bracketing + golden-section line search to choose τ minimizing 
E(u - τ g) at each iteration, and set u_{k+1} = u_k - τ g_k.
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
    DirichletBC,
    pi,
    RED,
    GREEN,
    conditional,
    gt,
    exp,
)
import argparse
import numpy.random as rand
import csv
import os


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


def gradient_descent_constrained(nx=128, degree=1, gamma=2**16,
                                 f_type="gaussian", f_center=(0.5, 0.5), f_width=2**-3, f_amp=1.0, f_shift=-2**-4,
                                 tau0=1e-2, max_iters=2**10,
                                 expand=2.0, max_bracket=20, golden_iters=25,
                                 ic_max=6, ic_amp=0, seed=None,
                                 output="gradient.pvd", save_every=1, energy_log=None,
                                 circular_inner_product=False):
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

    # Diagnostics
    E = lambda uu: assemble(energy_form(uu, f, gamma) * dx)

    # Output writer
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

    # Descent loop
    Lip = Constant(1)  # Exact Lipschitz value
    for k in range(1, max_iters + 1):
        # Compute gradient direction g
        g = gradient_direction(u, f, gamma, f_center, f_width, circular_inner_product)
        g_norm2 = assemble(dot(grad(g), grad(g)) * dx)

        if g_norm2 <= 1e-30:
            print(GREEN % f"iter {k}: gradient ~ 0, stopping.")
            break

        # Proper line search: bracket and golden-section minimize phi(tau)=E(u - tau g)
        def phi(tau, buf):
            buf.assign(u - float(tau) / Lip * g)
            bc.apply(buf)
            return E(buf)

        u_trial = Function(V)
        # Start with tau0; ensure descent
        tau_b = float(tau0)
        E_b = phi(tau_b, u_trial)
        shrink = 0
        while E_b >= E_curr and shrink < max_bracket and tau_b > 1e-16:
            tau_b *= 0.5
            E_b = phi(tau_b, u_trial)
            shrink += 1
        if E_b >= E_curr:
            print(GREEN % f"iter {k}: no descent found for current direction, stopping.")
            break

        # Expand to bracket minimum
        tau_left = 0.0
        E_left = E_curr
        tau_c = tau_b
        E_c = E_b
        for _ in range(max_bracket):
            tau_next = tau_c * float(expand)
            E_next = phi(tau_next, u_trial)
            if E_next > E_c:
                # Bracket [tau_left, tau_c, tau_next]
                a = tau_left
                d = tau_next
                break
            tau_left, E_left = tau_c, E_c
            tau_c, E_c = tau_next, E_next
        else:
            # If couldn't bracket, take best so far (tau_c)
            u.assign(u_trial)
            bc.apply(u)
            E_curr = E_c
            constraint_viol = conditional(gt(f - u, 0), f - u, 0)
            total_violation = assemble(constraint_viol * dx)
            print(GREEN % f"iter {k}: E={E_curr:.12e}, violation={total_violation:.12e}, tau={tau_c:.3e}, |g|_H1^2={g_norm2:.3e} (no-bracket)")
            
            # Record energy data
            energy_data.append({
                'iteration': k, 
                'energy': float(E_curr), 
                'violation': float(total_violation), 
                'tau': float(tau_c), 
                'gradient_norm': float(g_norm2)
            })
            
            if save_every and (k % save_every == 0):
                writer.write(u, time=float(k))
            continue

        # Golden-section search on [a, d]
        gr = (5**0.5 - 1) / 2.0
        x1 = d - gr * (d - a)
        x2 = a + gr * (d - a)
        f1 = phi(x1, u_trial)
        f2 = phi(x2, u_trial)
        for _ in range(golden_iters):
            if f1 > f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (d - a)
                f2 = phi(x2, u_trial)
            else:
                d = x2
                x2 = x1
                f2 = f1
                x1 = d - gr * (d - a)
                f1 = phi(x1, u_trial)
            if abs(d - a) <= 1e-12 * max(1.0, abs(a) + abs(d)):
                break

        if f1 < f2:
            tau_star, E_star = x1, f1
        else:
            tau_star, E_star = x2, f2

        # Accept step
        u.assign(u_trial)
        bc.apply(u)
        E_curr = E_star

        constraint_viol = conditional(gt(f - u, 0), f - u, 0)
        total_violation = assemble(constraint_viol * dx)
        print(GREEN % f"iter {k}: E={E_curr:.12e}, violation={total_violation:.12e}, tau={float(tau_star):.3e}, |g|_H1^2={g_norm2:.3e}")
        
        # Record energy data
        energy_data.append({
            'iteration': k, 
            'energy': float(E_curr), 
            'violation': float(total_violation), 
            'tau': float(tau_star), 
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
    p = argparse.ArgumentParser(description="H1-seminorm gradient descent for Gaussian-constrained optimization with Dirichlet BCs")
    p.add_argument("--nx", type=int, default=2**4, help="Number of elements (in each direction)")
    p.add_argument("--degree", type=int, default=2**2, help="Polynomial degree for u")
    p.add_argument("--gamma", type=float, default=2**16, help="Gradient penalty coefficient (interface width^2)")
    p.add_argument("--f-type", type=str, default="circular", choices=["gaussian", "circular"], help="Type of constraint function: gaussian or circular")
    p.add_argument("--f-center", type=float, nargs=2, default=[0.4, 0.7], help="Center (x,y) for constraint")
    p.add_argument("--f-width", type=float, default=2**-3, help="Width parameter (sigma for gaussian, radius for circular)")
    p.add_argument("--f-amp", type=float, default=1.0, help="Amplitude of gaussian constraint")
    p.add_argument("--f-shift", type=float, default=-2**-4, help="Vertical shift for gaussian constraint (negative shifts below plane)")
    p.add_argument("--tau0", type=float, default=2**7, help="Initial step size guess for line search")
    p.add_argument("--max-iters", type=int, default=2**10, help="Maximum iterations")
    p.add_argument("--expand", type=float, default=2.0, help="Bracketing expansion factor (>1)")
    p.add_argument("--max-bracket", type=int, default=20, help="Max bracketing steps")
    p.add_argument("--golden-iters", type=int, default=25, help="Golden-section iterations")
    p.add_argument("--ic-max", type=int, default=6, help="Max frequency index for A and B in sin(Aπx) and sin(Bπy)")
    p.add_argument("--ic-amp", type=float, default=0, help="Overall amplitude scaling for IC")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for IC")
    p.add_argument("--output", type=str, default="gradient.pvd", help="Output VTK/PVD filename for u trajectory")
    p.add_argument("--save-every", type=int, default=1, help="Save every N iterations")
    p.add_argument("--energy-log", type=str, default=None, help="CSV filename to save energy history (e.g., 'energies_linesearch.csv')")
    p.add_argument("--circular-inner-product", action="store_true", help="Use circular inner product (L2 on circle + H1-seminorm) instead of H1-seminorm")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gradient_descent_constrained(
        nx=args.nx,
        degree=args.degree,
        gamma=args.gamma,
        f_type=args.f_type,
        f_center=tuple(args.f_center),
        f_width=args.f_width,
        f_amp=args.f_amp,
        f_shift=args.f_shift,
        tau0=args.tau0,
        max_iters=args.max_iters,
        expand=args.expand,
        max_bracket=args.max_bracket,
        golden_iters=args.golden_iters,
        ic_max=args.ic_max,
        ic_amp=args.ic_amp,
        seed=args.seed,
        output=args.output,
        save_every=args.save_every,
        energy_log=args.energy_log,
        circular_inner_product=args.circular_inner_product,
    )
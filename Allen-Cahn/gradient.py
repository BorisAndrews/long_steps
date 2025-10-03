#!/usr/bin/env python3

"""
Gradient descent for the stationary Allen–Cahn energy under the H1 seminorm with
homogeneous Dirichlet boundary conditions (u = 0 on ∂Ω) and a fixed mean constraint.

We minimize
    E(u) = ∫ [ 1/4 (u^2 - 1)^2 + (1/(2γ)) |∇u|^2 ] dx

We compute the H1-seminorm gradient g at u by solving the Riesz representation with a zero-mean constraint:
    find (g, α) in H1_0 × R s.t. (∇g, ∇v) + α ∫ v dx + q ∫ g dx = dE/du[v]
so that ∫ g dx = 0 and updates preserve the mean. We then perform a bracketing + golden-section
line search to choose τ minimizing E(u - τ g) at each iteration, and set u_{k+1} = u_k - τ g_k.
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


def energy_form(u, gamma):
    # Allen–Cahn energy used in the Newton solver (with 1/gamma in front of |∇u|^2)
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

    # First variation for Allen–Cahn energy with 1/gamma gradient term
    L = ((u**2 - 1.0) * u) * v * dx + (1.0 / gamma_c) * dot(grad(u), grad(v)) * dx
    a = inner(grad(g), grad(v)) * dx + alpha * v * dx + q * g * dx

    w = Function(W)
    bc = DirichletBC(W.sub(0), 0.0, "on_boundary")
    problem = LinearVariationalProblem(a, L, w, bcs=bc)
    solver = LinearVariationalSolver(problem)
    solver.solve()
    g_sol, _alpha = w.subfunctions
    return g_sol


def gradient_descent(nx=128, degree=1, gamma=1e-2, m0=0.0,
                     tau0=1e-2, max_iters=2**10,
                     expand=2.0, max_bracket=20, golden_iters=25,
                     ic_max=6, ic_amp=2**-8, seed=None,
                     output="gradient.pvd", save_every=1):
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

    # Diagnostics
    E = lambda uu: assemble(energy_form(uu, gamma))

    # Output writer
    writer = VTKFile("output/" + output)
    writer.write(u, time=float(0))

    E_curr = E(u)
    mean_u = assemble(u * dx)
    print(GREEN % f"iter 0: E={E_curr:.12e}, mean(u)={mean_u:.12e}")

    # Descent loop
    for k in range(1, max_iters + 1):
        # Compute gradient direction g (mean-zero)
        g = gradient_direction(u, gamma)
        g_norm2 = assemble(dot(grad(g), grad(g)) * dx)

        if g_norm2 <= 1e-30:
            print(GREEN % f"iter {k}: gradient ~ 0, stopping.")
            break

        # Proper line search: bracket and golden-section minimize phi(tau)=E(u - tau g)
        def phi(tau, buf):
            buf.assign(u - float(tau) * g)
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
            mean_u = assemble(u * dx)
            print(GREEN % f"iter {k}: E={E_curr:.12e}, mean(u)={mean_u:.12e}, tau={tau_c:.3e}, |g|_H1^2={g_norm2:.3e} (no-bracket)")
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

        # Optional tiny mass-drift correction (not needed for Dirichlet BCs)
        # (left here for reference if adapting to other BCs)

        mean_u = assemble(u * dx)
        print(GREEN % f"iter {k}: E={E_curr:.12e}, mean(u)={mean_u:.12e}, tau={float(tau_star):.3e}, |g|_H1^2={g_norm2:.3e}")

        if save_every and (k % save_every == 0):
            writer.write(u, time=float(k))

    # Final save if not already saved at last iteration
    if (max_iters % save_every) != 0:
        writer.write(u, time=float(k))


def parse_args():
    p = argparse.ArgumentParser(description="H1-seminorm gradient descent for stationary Allen–Cahn energy with Dirichlet BCs and mass constraint")
    p.add_argument("--nx", type=int, default=2**4, help="Number of elements (in each direction)")
    p.add_argument("--degree", type=int, default=2**2, help="Polynomial degree for u")
    p.add_argument("--gamma", type=float, default=2**9, help="Gradient penalty coefficient (interface width^2)")
    p.add_argument("--m0", type=float, default=0.0, help="Target mean (mass constraint)")
    p.add_argument("--tau0", type=float, default=2**7, help="Initial step size guess for line search")
    p.add_argument("--max-iters", type=int, default=2**10, help="Maximum iterations")
    p.add_argument("--expand", type=float, default=2.0, help="Bracketing expansion factor (>1)")
    p.add_argument("--max-bracket", type=int, default=20, help="Max bracketing steps")
    p.add_argument("--golden-iters", type=int, default=25, help="Golden-section iterations")
    p.add_argument("--ic-max", type=int, default=6, help="Max frequency index for A and B in sin(Aπx) and sin(Bπy)")
    p.add_argument("--ic-amp", type=float, default=2**-8, help="Overall amplitude scaling for IC")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for IC")
    p.add_argument("--output", type=str, default="gradient.pvd", help="Output VTK/PVD filename for u trajectory")
    p.add_argument("--save-every", type=int, default=1, help="Save every N iterations")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gradient_descent(
        nx=args.nx,
        degree=args.degree,
        gamma=args.gamma,
        m0=args.m0,
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
    )

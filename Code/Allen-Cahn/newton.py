#!/usr/bin/env python3

"""
Compute stationary states of the Allen–Cahn equation by minimizing the free-energy

    E(u) = ∫_Ω [ 1/4 (u^2 - 1)^2 + (1/(2γ)) |∇u|^2 ] dx

with homogeneous Dirichlet boundary conditions (u = 0 on ∂Ω) and a fixed mean (mass) constraint:

    (1/|Ω|) ∫_Ω u dx = m0.

We enforce the mean constraint via a Lagrange multiplier λ in a mixed formulation.
"""

from firedrake import (
    UnitSquareMesh,
    FunctionSpace,
    Function,
    TestFunctions,
    TrialFunction,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    Constant,
    split,
    derivative,
    assemble,
    grad,
    dot,
    dx,
    VTKFile,
    SpatialCoordinate,
    sin,
    RED,
    GREEN,
    BLUE,
    DirichletBC,
    pi,
)
import argparse
import numpy.random as rand

def solve_stationary(nx=2**7, degree=2, gamma=2**10, m0=0.0,
                     ic_max=6, ic_amp=2**0, seed=None,
                     output="newton.pvd"):
    # Mesh and function spaces
    mesh = UnitSquareMesh(nx, nx, quadrilateral=True)
    V = FunctionSpace(mesh, "Q", degree)
    R = FunctionSpace(mesh, "R", 0)
    W = V * R
    print(RED % f"Degrees of freedom: {V.dim()}")

    # Unknowns and tests
    w = Function(W, name="(u, lambda)")
    u, lam = split(w)
    v, q = TestFunctions(W)

    gamma_c = Constant(gamma)
    m0_c = Constant(m0)

    # Energy density and total energy form E(u)
    psi = 0.25 * (u**2 - 1.0)**2 + 0.5 / gamma_c * dot(grad(u), grad(u))
    E_form = psi * dx

    # Mixed residual: dE/du(v) + λ ∫ v dx = 0 ; and ∫ (u - m0) q dx = 0
    F = derivative(E_form, u, v) + lam * v * dx + (u - m0_c) * q * dx

    # Jacobian
    dw = TrialFunction(W)
    J = derivative(F, w, dw)

    # Initial guess: sum over all sine modes sin(Aπx)sin(Bπy) up to max A,B (Dirichlet-compatible)
    x, y = SpatialCoordinate(mesh)
    u0 = Function(V, name="u")
    rng = rand.RandomState(seed) if seed is not None else rand
    base = 0
    for Ax in range(1, int(ic_max) + 1):
        for By in range(1, int(ic_max) + 1):
            coeff = float(rng.standard_normal())
            base = base + coeff * sin(Ax * pi * x) * sin(By * pi * y) / (Ax**2 + By**2)**1.5
    u0.interpolate(ic_amp * base)

    # Problem and solver
    bcs = DirichletBC(W.sub(0), 0, "on_boundary")
    problem = NonlinearVariationalProblem(F, w, J=J, bcs=bcs)
    solver = NonlinearVariationalSolver(problem)

    # Report initial energy and mass
    # Assign initial guess into mixed Function
    w.sub(0).assign(u0)
    w.sub(1).assign(0.0)
    area = assemble(Constant(1.0) * dx(domain=mesh))
    E0 = assemble(E_form)
    mass0 = assemble(u * dx) / area
    print(GREEN % f"Initial:  E = {E0:.12e},  mean(u) = {mass0:.12e}")

    # Solve
    solver.solve()

    u_sol, lam_sol = w.subfunctions
    u_sol.rename("u")  # Name the solution variable 'u' for consistency

    # Diagnostics
    E_final = assemble((0.25 * (u_sol**2 - 1.0)**2 + 0.5 / gamma_c * dot(grad(u_sol), grad(u_sol))) * dx)
    mean_u = assemble(u_sol * dx) / area
    lam_value = lam_sol.dat.data[0] if lam_sol.function_space().ufl_element().family() == "Real" else float("nan")
    print(GREEN % f"Final:    E = {E_final:.12e},  mean(u) = {mean_u:.12e},  lambda ≈ {lam_value:.12e}")

    # Save solution
    if output:
        print(BLUE % f"Writing solution to output/{output}")
        VTKFile("output/" + output).write(u_sol)


def parse_args():
    p = argparse.ArgumentParser(description="Stationary Allen–Cahn solver (energy minimization with mass constraint and Dirichlet BCs)")
    p.add_argument("--nx", type=int, default=2**4, help="Number of elements (in each direction)")
    p.add_argument("--degree", type=int, default=2**2, help="Polynomial degree for u")
    p.add_argument("--gamma", type=float, default=2**9, help="Gradient penalty coefficient (1/interface width^2)")
    p.add_argument("--m0", type=float, default=0.0, help="Target mean (mass constraint)")
    p.add_argument("--ic-max", type=int, default=6, help="Max frequency index for A and B in sin(Aπx) and sin(Bπy)")
    p.add_argument("--ic-amp", type=float, default=2**0, help="Overall amplitude scaling for IC")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for IC")
    p.add_argument("--output", type=str, default="newton.pvd", help="Output VTK/PVD filename for u")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    solve_stationary(
        nx=args.nx,
        degree=args.degree,
        gamma=args.gamma,
        m0=args.m0,
        ic_max=args.ic_max,
        ic_amp=args.ic_amp,
        seed=args.seed,
        output=args.output,
    )

#!/usr/bin/env python3

"""
Compute stationary states for the constrained optimization problem by minimizing:

    E(u) = ∫_Ω max(f - u, 0) dx + (1/(2γ)) ∫_Ω |∇u|^2 dx

with homogeneous Dirichlet boundary conditions (u = 0 on ∂Ω).

The max(f - u, 0) term is handled using conditional expressions in UFL.
"""

from firedrake import (
    UnitSquareMesh,
    FunctionSpace,
    Function,
    TestFunction,
    TrialFunction,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    Constant,
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
    conditional,
    gt,
    exp,
)
import argparse
import numpy.random as rand

def solve_constrained(nx=2**7, degree=2, gamma=2**9,
                      f_type="gaussian", f_center=(0.5, 0.5), f_width=2**-3, f_amp=1.0, f_shift=-2**-4,
                      ic_max=6, ic_amp=0, seed=None,
                      output="newton.pvd"):
    # Mesh and function spaces
    mesh = UnitSquareMesh(nx, nx, quadrilateral=True)
    V = FunctionSpace(mesh, "Q", degree)
    print(RED % f"Degrees of freedom: {V.dim()}")

    # Unknown and test
    u = Function(V, name="u")
    v = TestFunction(V)

    gamma_c = Constant(gamma)

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

    # Energy density: max(f - u, 0) + (1/(2γ)) |∇u|^2
    # Use conditional to handle max(f - u, 0)
    constraint_violation = conditional(gt(f - u, 0), f - u, 0)
    psi = constraint_violation + 0.5 / gamma_c * dot(grad(u), grad(u))
    E_form = psi * dx

    # Residual: dE/du(v) = 0
    F = derivative(E_form, u, v)

    # Jacobian
    du = TrialFunction(V)
    J = derivative(F, u, du)

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

    # Problem and solver
    bcs = DirichletBC(V, 0, "on_boundary")
    problem = NonlinearVariationalProblem(F, u, J=J, bcs=bcs)
    solver = NonlinearVariationalSolver(problem)

    # Report initial energy
    E0 = assemble(E_form)
    print(GREEN % f"Initial:  E = {E0:.12e}")

    # Solve
    solver.solve()

    u.rename("u")  # Name the solution variable 'u' for consistency

    # Diagnostics
    constraint_viol_final = conditional(gt(f - u, 0), f - u, 0)
    E_final = assemble((constraint_viol_final + 0.5 / gamma_c * dot(grad(u), grad(u))) * dx)
    total_violation = assemble(constraint_viol_final * dx)
    print(GREEN % f"Final:    E = {E_final:.12e},  total_violation = {total_violation:.12e}")

    # Save solution and constraint function
    if output:
        print(BLUE % f"Writing solution to output/{output}")
        # Also save the constraint function f for visualization
        f_func = Function(V, name="f")
        f_func.interpolate(f)
        constraint_func = Function(V, name="constraint_violation")
        constraint_func.interpolate(constraint_viol_final)
        VTKFile("output/" + output).write(u, f_func, constraint_func)


def parse_args():
    p = argparse.ArgumentParser(description="Gaussian-constrained optimization solver (minimize max(f-u,0) + H1 seminorm with Dirichlet BCs)")
    p.add_argument("--nx", type=int, default=2**4, help="Number of elements (in each direction)")
    p.add_argument("--degree", type=int, default=2**2, help="Polynomial degree for u")
    p.add_argument("--gamma", type=float, default=2**9, help="Gradient penalty coefficient (1/interface width^2)")
    p.add_argument("--f-type", type=str, default="circular", choices=["gaussian", "circular"], help="Type of constraint function: gaussian or circular")
    p.add_argument("--f-center", type=float, nargs=2, default=[0.4, 0.7], help="Center (x,y) for constraint")
    p.add_argument("--f-width", type=float, default=2**-3, help="Width parameter (sigma for gaussian, radius for circular)")
    p.add_argument("--f-amp", type=float, default=1.0, help="Amplitude of gaussian constraint")
    p.add_argument("--f-shift", type=float, default=-2**-4, help="Vertical shift for gaussian constraint (negative shifts below plane)")
    p.add_argument("--ic-max", type=int, default=6, help="Max frequency index for A and B in sin(Aπx) and sin(Bπy)")
    p.add_argument("--ic-amp", type=float, default=0, help="Overall amplitude scaling for IC")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for IC")
    p.add_argument("--output", type=str, default="newton.pvd", help="Output VTK/PVD filename for u")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    solve_constrained(
        nx=args.nx,
        degree=args.degree,
        gamma=args.gamma,
        f_type=args.f_type,
        f_center=tuple(args.f_center),
        f_width=args.f_width,
        f_amp=args.f_amp,
        f_shift=args.f_shift,
        ic_max=args.ic_max,
        ic_amp=args.ic_amp,
        seed=args.seed,
        output=args.output,
    )
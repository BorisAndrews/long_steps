# Constrained Optimization Solvers (Firedrake)

This folder provides Firedrake scripts for solving constrained optimization problems by minimizing:

E(u) = ∫ max(f - u, 0) dx + (1/(2γ)) ∫ |∇u|² dx

with homogeneous Dirichlet boundary conditions (u = 0 on ∂Ω), where f(x,y) is a given constraint function and we enforce u ≤ f pointwise via the max penalty.

## Files

- `newton.py`: Newton solver for constrained optimization
- `gradient.py`: Gradient descent with line search for constrained optimization  
- `gradient_periodic.py`: Gradient descent with periodic step sizes for constrained optimization

## Usage

Requires [Firedrake](https://www.firedrakeproject.org/). Activate your Firedrake environment, then run:

### Examples

```bash
python3 newton.py --nx 128 --gamma 1e-2 --f-type gaussian --f-center 0.5 0.5 --f-width 0.2 --output constrained_newton.pvd
python3 gradient.py --nx 128 --gamma 1e-2 --f-type gaussian --f-center 0.5 0.5 --f-width 0.2 --output constrained_gradient.pvd
python3 gradient_periodic.py --nx 128 --gamma 1e-2 --f-type gaussian --taus "1e-2,5e-3,1e-2" --output constrained_periodic.pvd
```

### Key Options

- `--nx`: mesh resolution on the unit square
- `--degree`: CG polynomial degree for u (default 1)
- `--gamma`: gradient penalty coefficient (1/interface width²)
- `--f-type`: type of constraint function f (gaussian, sine, constant)
- `--f-center`: center coordinates for gaussian constraint
- `--f-width`: width parameter for gaussian constraint
- `--f-amp`: amplitude of constraint function f
- `--taus`: comma-separated step sizes for periodic gradient descent
- `--cycle-length`: predefined step size cycle (3, 7, 15, 31, 63, 127)

### Initial Conditions

All scripts use randomized initial conditions as a sum over sine modes:

u₀(x,y) = (ic_amp / √(ic_max_kx × ic_max_ky)) × Σ(A=1 to ic_max_kx) Σ(B=1 to ic_max_ky) c_{A,B} sin(Aπx) sin(Bπy)

Control via `--ic-max-kx`, `--ic-max-ky`, `--ic-amp`, and `--seed`.

## Constraint Functions

The constraint function f(x,y) can be:

- **Gaussian**: f = A·exp(-((x-cx)²+(y-cy)²)/(2σ²))
- **Sine**: f = A·sin(2πx)sin(2πy) 
- **Constant**: f = A

## Output

All scripts write VTK files (PVD format) with:
- Solution u
- Constraint function f  
- Constraint violation max(f-u,0)

For gradient descent methods, the output contains a time series over iterations.

## Mathematical Background

The energy functional penalizes constraint violations where u > f and includes an H1 penalty term for regularity. The max(f-u,0) term introduces non-smoothness that is handled using UFL conditional expressions.

For gradient descent methods, the H1-seminorm gradient is computed via Riesz representation:
find g ∈ H₁₀ such that (∇g, ∇v) = δE/δu[v]

The Newton method solves the Euler-Lagrange equation directly using automatic differentiation.

## Notes

- The max(f-u,0) term introduces non-smoothness handled with UFL conditionals
- Solutions enforce u ≤ f pointwise through the penalty formulation
- Different constraint functions lead to different optimal solution patterns
- Gradient descent uses bracketing line search (gradient.py) or fixed periodic steps (gradient_periodic.py)

# Allen-Cahn Energy Minimization Solvers (Firedrake)

This folder provides Firedrake scripts that compute stationary states of the Allen–Cahn equation by minimizing the free energy:

E(u) = ∫ [ 1/4 (u² - 1)² + (1/(2γ)) |∇u|² ] dx

with a fixed mean constraint ∫ u dx = m0 |Ω| using a Lagrange multiplier and homogeneous Dirichlet boundary conditions (u = 0 on ∂Ω).

## Files

- `newton.py`: Newton solver for Allen-Cahn energy minimization
- `gradient.py`: Gradient descent with line search for Allen-Cahn
- `gradient_periodic.py`: Gradient descent with periodic step sizes for Allen-Cahn

## Usage

Requires [Firedrake](https://www.firedrakeproject.org/). Activate your Firedrake environment, then run:

### Examples

```bash
python3 newton.py --nx 128 --gamma 1e-2 --m0 0.0 --output newton.pvd
python3 gradient.py --nx 128 --gamma 1e-2 --m0 0.0 --output gradient.pvd
python3 gradient_periodic.py --nx 128 --gamma 1e-2 --taus "1e-2,5e-3,1e-2" --output gradient_periodic.pvd
```

### Key Options

- `--nx`: mesh resolution on the unit square
- `--degree`: CG polynomial degree for u (default 1)
- `--gamma`: gradient penalty coefficient (1/interface width²)
- `--m0`: target mean value for u (mass constraint)
- `--tau0`: initial step size guess for line search (gradient.py)
- `--taus`: comma-separated step sizes for periodic gradient descent
- `--cycle-length`: predefined step size cycle (3, 7, 15, 31, 63, 127)

### Initial Conditions

All scripts use randomized initial conditions as a sum over sine modes:

u₀(x,y) = (ic_amp / √(ic_max_kx × ic_max_ky)) × Σ(A=1 to ic_max_kx) Σ(B=1 to ic_max_ky) c_{A,B} sin(Aπx) sin(Bπy)

Control via `--ic-max-kx`, `--ic-max-ky`, `--ic-amp`, and `--seed`.

## Output

All scripts write VTK files (PVD format) with the solution `u` for visualization in ParaView.

For gradient descent methods, the output contains a time series over iterations showing the evolution toward the stationary state.

## Mathematical Background

The Allen-Cahn energy functional models phase separation with:
- A double-well potential 1/4(u²-1)² favoring u ≈ ±1
- A gradient penalty (1/(2γ))|∇u|² promoting smooth interfaces
- The parameter γ controls interface width (smaller γ = sharper interfaces)

The Euler–Lagrange equations are assembled as mixed variational problems in (u, λ) with λ as a Lagrange multiplier enforcing the mean constraint ∫ u dx = m0|Ω|.

For gradient descent methods, the H1-seminorm gradient is computed via Riesz representation with zero-mean constraint:
find (g, α) ∈ H₁₀ × ℝ such that (∇g, ∇v) + α ∫ v dx + q ∫ g dx = δE/δu[v]

This ensures ∫ g dx = 0 so that updates preserve the mass constraint.

## Notes

- Depending on parameters and initial guess, solvers may converge to different local minima (phase-separated patterns)
- Vary γ and mesh resolution to explore different interface structures
- The Newton solver typically converges faster but may be less robust for challenging initial conditions
- Gradient descent with line search is more robust but slower
- Periodic gradient descent can accelerate convergence with well-tuned step size sequences
- For larger problems, consider changing the linear solver preconditioner from LU to an iterative method

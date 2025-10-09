# READING GROUP PLAN

## MOTIVATION (WHEN NEWTON FAILS)

we're used to using Newton, and who can blame us (when it works, the convergence rates are practically unbeatable)

it's not always a good idea though:

1. Newton-Kantorovich has some pretty strict criteria though (your residual must be (locally) Lipschitz differentiable + invertible at the minimiser)

    it's not hard to construct a case where this isn't true

    exemplified in this little obstacle problem demo with functional `E'` (Nemystkii operator)

    (if my calculations are correct...) this functional's derivative `E'` is differentiable, but not continuously differentiable, never mind Lipschitz differentiable

    no surprise then, we find numerically that Newton fails

    N.B. semismooth Newton

2. if we're trying to do a very high order, the nonlinearity makes this super costly to assemble

3. for nonlinear problem with many solutions, we often just don't start in an attractor (I mean you can do things like continuation, but still)

## BACK TO BASICS (GRADIENT DESCENT)

say then that we're thinking about an energy minimisation problem then

let me remind you all of the ol' reliable: gradient descent (GD)

you might not have seen GD in the function space setting before, but the idea's identical

1. *for convergence, this step size must be in `(0, 2L)`, where `L` is the Lipschitz constant of `E'` (the proof is simple)*

    *the immediate consequence of this is that we need one lower level of regularity to guarantee convergence*

2. *the linear operator to invert is just our space's canonical inner product, and will be identical every time*

    *even at high order, a nice choice of basis makes this easy to invert*

3. *the convergence of GD is guaranteed, no matter where you start*

## 2D DEMO

let's show you that step size result

we see as expected, that step sizes outside `(0, 2L)` break convergence (this bound is, apparently, tight)

we could tweak inside that range to get the fastest convergence rate possible...

*...but, even better, let's run linesearch, to get some idea maybe of what rate we should pick*

oh that's interesting...

*...linesearch isn't taking uniform step sizes, but alternating between a small one and a large one (in fact, so large that it would normally be forbidden!)*

this gives us an idea:

*if we use periodic patterns of step sizes, we can accelerate the convergence!*

## A NOTE ON THE CONVERGENCE RATES OF GD

don't be deceived from this figure into thinking that GD converges at a rate `O(const.^-n)`

that exponential rate relies on something called "strong convexity" (with definition and diagram here)...

*...that constant `C` is dependent on the strong convexity coefficient...*

*...and sure, that doesn't in general hold (e.g. `x^4 + y^2`)...*

*...but more importantly for us, that strong convexity coefficient might depend very poorly on the problem parameters*

a robust convergence rate for CG can be given by looking not at the convergence of `x` (or `u`), but at the convergence of `E`

this makes sense in application:

*if we're doing an energy minimisation problem, the key point is minimising the energy, not finding the functional that minimises the energy (slightly different perspectives)*

anyway, we can get convergence in `E` like `O(1/n)`

sure, this is robust...

*...but it's really, really shit (numerical demo)*

let's see if we can do better through what we learnt about alternating step sizes

# VIASM Codes for Summer School on ANM 2025

Hands-on codes for [VIASM 2025 summer school on advanced numerical methods for deterministic and stochastic differential equations](https://viasm.edu.vn/en/hdkh/summer-school-on-anm2025).

These codes require a modern Python installation.

Codes are grouped according to type:

* `initial_demo` -- simple demonstration scripts showing the use of Python for mathematical calculations and plotting.
* `shared` -- reusable `ImplicitSolver` class, to be used by implicit ODE methods.
* `newton` -- test driver to show use of `ImplicitSolver`.
* `forward_euler` -- simple IVP "evolution" routine, based on the simplest IVP solver.  Basic approach for timestep adaptivity.  Contains two classes, `ForwardEuler` (fixed-step evolution) and `AdaptEuler` (adaptive-step evolution).
* `simple_implicit` -- simple implicit ODE solver classes, `BackwardEuler` and `Trapezoidal`, showing use of the `ImplicitSolver` class for implicit ODE methods.
* `explicit_one_step` -- higher-order explicit, one-step, ODE integration methods, containing the `Taylor2` and `ERK` classes.
* `implicit_one_step` -- higher-order implicit, one-step, ODE integration methods, containing the `DIRK` and `IRK` classes.

Daniel R. Reynolds
[Mathematics @ SMU](https://www.smu.edu/dedman/academics/departments/math)
[Mathematics and Statistics @ UMBC](https://mathstat.umbc.edu)

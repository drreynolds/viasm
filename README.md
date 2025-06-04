# VIASM Codes for Summer School on ANM 2025

Hands-on codes for [VIASM 2025 summer school on advanced numerical methods for deterministic and stochastic differential equations](https://viasm.edu.vn/en/hdkh/summer-school-on-anm2025).

## Dependencies

These codes require a modern Python installation.  All dependencies are included in the file `python_requirements.txt`.

*Installation note:* if you need to install these, it is generally recommended to install Python dependencies in a virtual environment; in recent versions of Python, you can create an activate a virtual environment via

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Following this, the Python dependencies can be installed via

```bash
pip install -r python_requirements.txt
```

## File layout

Classes and user-callable functions are in files that begin with capital letters, while scripts that run various demos are in files that begin with lower-case letters.  These are grouped as follows:

### Background

Basic Python demos:

* `numpy_demo.py` -- simple script showing Numpy usage.
* `plotting_demo.py` -- simple script showing how to generate plots.

### Fixed-step Runge--Kutta methods and implicit solvers

* `ForwardEuler.py` -- simple baseline explicit IVP time-stepper class.
* `ERK.py` -- explicit Runge--Kutta IVP time-stepper class.
* `driver_explicit_fixed.py` -- script to test explicit fixed-step methods.
* `driver_explicit_stability.py` -- script to demonstrate stability limitations of explicit methods.
* `ImplicitSolver.py` -- reusable nonlinear solver class for implicit IVP methods.
* `BackwardEuler.py` -- simple baseline implicit IVP time-stepper class.
* `DIRK.py` -- diagonally-implicit Runge--Kutta IVP time-stepper class.
* `driver_implicit_fixed.py` -- script to test implicit fixed-step methods.

### Adaptive-step Runge--Kutta methods

* `AdaptERK.py` -- explicit Runge--Kutta adaptive IVP solver class.
* `driver_explicit_adaptive.py` -- script to test explicit adaptive-step methods.
* `AdaptDIRK.py` -- implicit Runge--Kutta adaptive IVP solver class.
* `driver_implicit_adaptive.py` -- script to test implicit adaptive-step methods.
* `driver_adaptive_timescale.py` -- script to demonstrate how adaptive solvers track dynamical time scales.
* `driver_adaptive_stability.py` -- script to demonstrate how adaptive solvers can assess stiffness.

### Multirate methods

* `LieTrotterSubcycling.py` -- simple fixed-step Lie--Trotter subcycling IVP time-stepper class.
* `MRI.py` -- higher-order fixed-step multirate infinitesimal (MRI)ling IVP time-stepper class.
* `driver_subcycling.py` -- script to demonstrate accuracy differences for various multirate methods.

### Auxiliary utilities

* `RK_stability.py` -- function to plot linear stability regions for Runge--Kutta methods.

## Authors

[Daniel R. Reynolds](https://drreynolds.github.io/)  
[Mathematics @ SMU](https://www.smu.edu/dedman/academics/departments/math)  
[Mathematics & Statistics @ UMBC](https://mathstat.umbc.edu)

[Van Hoang Nguyen](https://www.depts.ttu.edu/math/facultystaff)  
[Mathematics & Statistics @ TTU](https://www.depts.ttu.edu/math)

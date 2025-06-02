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

Classes and user-callable functions are in files that begin with capital letters:

* `AdaptERK.py` -- explicit Runge--Kutta adaptive IVP solver class.
* `BackwardEuler.py` -- simple baseline implicit IVP time-stepper class.
* `DIRK.py` -- diagonally-implicit Runge--Kutta IVP time-stepper class.
* `ERK.py` -- explicit Runge--Kutta IVP time-stepper class.
* `ForwardEuler.py`
-- simple baseline explicit IVP time-stepper class.
* `ImplicitSolver.py` -- reusable nonlinear solver class for implicit IVP methods.
* `RK_stability.py` -- function to generate linear stability plots for Runge--Kutta methods.

Scripts that run various various demos are in files that begin with lower-case letters:

* `driver_adaptERK.py` --
* `driver_DIRK_system.py` --
* `driver_DIRK.py` --
* `driver_ERK.py` --
* `numpy_demo.py` --
* `plotting_demo.py` --
* `stability_experiment.py` --
* `test_implicit_solver.py` --

## Authors

[Daniel R. Reynolds](https://drreynolds.github.io/)  
[Mathematics @ SMU](https://www.smu.edu/dedman/academics/departments/math)  
[Mathematics and Statistics @ UMBC](https://mathstat.umbc.edu)

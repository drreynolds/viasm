#!/usr/bin/env python3
#
# Script that runs various adaptive implicit methods on the Oregonator problem,
# including built-in adaptive solvers.
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University

import numpy as np
import matplotlib.pyplot as plt
from ImplicitSolver import *
from AdaptDIRK import *

# Oregonator initial condition and parameters
y0 = np.array([5.025e-11, 6e-7, 7.236e-8], dtype=float)
t0 = 0.0
tf = 360.0
k1 = 2.57555802e8
k2 = 7.72667406e1
k3 = 1.28777901e7
k4 = 1.29421790e-2
k5 = 1.60972376e-1

# Oregonator functions
def f(t,y):
    """
    Right-hand side function, f(t,y), for the IVP.
    """
    return np.array([ -k1*y[0]*y[1] + k2*y[0] - k3*y[0]*y[0] + k4*y[1],
                      -k1*y[0]*y[1] - k4*y[1] + k5*y[2],
                      k2*y[0] - k5*y[2] ], dtype=float)

def J(t,y):
    """
    Jacobian (in dense matrix format) of the right-hand side
    function, J(t,y) = df/dy, for the IVP.
    """
    return np.array([[ -k1*y[1] + k2 - 2.0*k3*y[0], -k1*y[0] + k4, 0 ],
                     [ -k1*y[1], -k1*y[0] - k4, k5 ],
                     [ k2, 0, -k5 ]], dtype=float)

def reference_solution(N, reltol=1e-8, abstol=[1e-16, 1e-20, 1e-18]):
    """
    Function that returns a high-accuracy reference solution to
    the IVP over a specified number of time outputs -- both the
    array of these time outputs and the solution at these outputs
    are returned.
    """
    from scipy.integrate import solve_ivp
    tvals = np.linspace(t0, tf, N)
    ivpsol = solve_ivp(f, (t0,tf), y0, method='BDF', jac=J,
                       t_eval=tvals, rtol=reltol, atol=abstol)
    if (not ivpsol.success):
        raise Exception("Failed to generate reference solution")
    return (tvals, ivpsol.y)

# shared testing data
interval = tf - t0
Nout = 100
tspan,yref = reference_solution(Nout+1, reltol=1e-12)
hvals = interval/Nout/np.array([2000, 3000, 4000, 5000], dtype=float)
solver = ImplicitSolver(J, solver_type='dense', maxiter=20,
                        rtol=1e-9, atol=1e-12, Jfreq=3)

# tolerances
rtol = 1.e-6
atol = 1.e-12

# create adaptive DIRK solvers
AD21 = AdaptDIRK(f, y0, solver, SDIRK21(), rtol=rtol, atol=atol, save_step_hist=True)
AD32 = AdaptDIRK(f, y0, solver, ESDIRK32(), rtol=rtol, atol=atol, save_step_hist=True)
AD43 = AdaptDIRK(f, y0, solver, ESDIRK43(), rtol=rtol, atol=atol, save_step_hist=True)
AD54 = AdaptDIRK(f, y0, solver, ESDIRK54(), rtol=rtol, atol=atol, save_step_hist=True)

# adaptive tests
print("\nAdaptive DIRK21 solver:")
Y_AD21, success = AD21.Evolve(tspan, y0)
step_hist_AD21 = AD21.get_step_history()
err_AD21 = np.linalg.norm(Y_AD21 - np.transpose(yref), 1)
print("  steps = %5i  fails = %2i, error = %.2e\n" %
      (AD21.get_num_steps(), AD21.get_num_error_failures(), err_AD21))
solver.reset()

print("\nAdaptive DIRK32 solver:")
Y_AD32, success = AD32.Evolve(tspan, y0)
step_hist_AD32 = AD32.get_step_history()
err_AD32 = np.linalg.norm(Y_AD32 - np.transpose(yref), 1)
print("  steps = %5i  fails = %2i, error = %.2e\n" %
      (AD32.get_num_steps(), AD32.get_num_error_failures(), err_AD32))
solver.reset()

print("\nAdaptive DIRK43 solver:")
Y_AD43, success = AD43.Evolve(tspan, y0)
step_hist_AD43 = AD43.get_step_history()
err_AD43 = np.linalg.norm(Y_AD43 - np.transpose(yref), 1)
print("  steps = %5i  fails = %2i, error = %.2e\n" %
      (AD43.get_num_steps(), AD43.get_num_error_failures(), err_AD43))
solver.reset()

print("\nAdaptive DIRK54 solver:")
Y_AD54, success = AD54.Evolve(tspan, y0)
step_hist_AD54 = AD54.get_step_history()
err_AD54 = np.linalg.norm(Y_AD54 - np.transpose(yref), 1)
print("  steps = %5i  fails = %2i, error = %.2e\n" %
      (AD54.get_num_steps(), AD54.get_num_error_failures(), err_AD54))
solver.reset()

# create plots for adaptive runs
plt.figure()
plt.plot(tspan, Y_AD21)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.title('DIRK21 Solution')
plt.savefig('adaptive_dirk21.png')

plt.figure()
plt.plot(tspan, Y_AD32)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.title('DIRK32 Solution')
plt.savefig('adaptive_dirk32.png')

plt.figure()
plt.plot(tspan, Y_AD43)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.title('DIRK43 Solution')
plt.savefig('adaptive_dirk43.png')

plt.figure()
plt.plot(tspan, Y_AD54)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.title('DIRK54 Solution')
plt.savefig('adaptive_dirk54.png')

plt.figure()
plt.plot(step_hist_AD21['t'], step_hist_AD21['h'], 'b-', label='DIRK21')
plt.plot(step_hist_AD32['t'], step_hist_AD32['h'], 'k-', label='DIRK32')
plt.plot(step_hist_AD43['t'], step_hist_AD43['h'], 'g-', label='DIRK43')
plt.plot(step_hist_AD54['t'], step_hist_AD54['h'], 'm-', label='DIRK54')
for i in range(len(step_hist_AD21['t'])):
    if (step_hist_AD21['err'][i] > 1.0):
        plt.plot(step_hist_AD21['t'][i], step_hist_AD21['h'][i], 'bx')
for i in range(len(step_hist_AD32['t'])):
    if (step_hist_AD32['err'][i] > 1.0):
        plt.plot(step_hist_AD32['t'][i], step_hist_AD32['h'][i], 'kx')
for i in range(len(step_hist_AD43['t'])):
    if (step_hist_AD43['err'][i] > 1.0):
        plt.plot(step_hist_AD43['t'][i], step_hist_AD43['h'][i], 'gx')
for i in range(len(step_hist_AD54['t'])):
    if (step_hist_AD54['err'][i] > 1.0):
        plt.plot(step_hist_AD54['t'][i], step_hist_AD54['h'][i], 'mx')
plt.xlabel('$t$')
plt.ylabel('$h$')
plt.title('Adaptive step history')
plt.legend()
plt.savefig('adaptive_DIRK_steps.png')

plt.show()

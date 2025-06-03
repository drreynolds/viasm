#!/usr/bin/env python3
#
# Script that runs various adaptive methods on the nonlinear Kvaerno
# Prothero and Robinson problem:
#    [u]' = [ G  e ] [(-1+u^2-r)/(2u)] + [      r'(t)/(2u)        ]
#    [v]    [ e -1 ] [(-2+v^2-s)/(2v)]   [ s'(t)/(2*sqrt(2+s(t))) ]
# where r(t) = 0.5*cos(t),  s(t) = cos(w*t),  0 < t < 5.
# This problem has analytical solution given by
#    u(t) = sqrt(1+r(t)),  v(t) = sqrt(2+s(t)).
#
# We use the parameters:
#   e = inter-variable coupling strength (0.5)
#   G = stiffness at slow time scale (varies)
#   w = variable time-scale separation factor (10)
#
# This script uses adaptive explicit and implicit solvers to assess problem
# stiffness as G is varied.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
import matplotlib.pyplot as plt
from ImplicitSolver import *
from AdaptDIRK import *
from AdaptERK import *

# KPR problem parameters
Tf = 5
Nt = 50
tvals = np.linspace(0, Tf, Nt+1)
e = 0.5
w = 10
G = [-1, -10, -100, -1000, -10000]  # stiffness values to test

# KPR component functions
def r(t):
    return 0.5*np.cos(t)
def s(t):
    return np.cos(w*t)
def rdot(t):
    return -0.5*np.sin(t)
def sdot(t):
    return -w*np.sin(w*t)

# KPR true solution functions
def utrue(t):
    return np.sqrt(1+r(t))
def vtrue(t):
    return np.sqrt(2+s(t))
def ytrue(t):
    return np.array((utrue(t), vtrue(t)))

# initial condition
Y0 = ytrue(0)

# true solution at each output time
Ytrue = np.zeros((2,Nt+1))
for i in range(Nt+1):
  Ytrue[:,i] = ytrue(tvals[i])

# loop over stiffness values
for g in G:
    print("\nKPR problem with G = %i\n" % (g))

    # KPR right-hand side function
    def f(t, y):
        u = y[0]
        v = y[1]
        return (np.array([[g, e], [e, -1]])
                @ np.array([(-1 + u**2 - r(t)) / (2 * u),
                            (-2 + v**2 - s(t)) / (2 * v)])
                + np.array([rdot(t) / (2 * u), sdot(t) / (2 * np.sqrt(2 + s(t)))]))

    # KPR Jacobian function
    def J(t, y):
        u = y[0]
        v = y[1]
        return np.array([[g/2 + (g*(1+r(t))+rdot(t))/(2*u**2),
                             e/2+e*(2+s(t))/(2*v**2)],
                         [e/2+e*(1+r(t))/(2*u**2), -1/2 - (2+s(t))/(2*v**2)]])

    solver = ImplicitSolver(J, solver_type='dense', maxiter=20,
                            rtol=1e-9, atol=1e-12, Jfreq=3)

    # tolerances
    rtol = 1.e-3
    atol = 1.e-11

    # create adaptive ERK and DIRK solvers
    E32 = AdaptERK(f, Y0, ERK32(), rtol=rtol, atol=atol, save_step_hist=True)
    D32 = AdaptDIRK(f, Y0, solver, ESDIRK32(), rtol=rtol, atol=atol, save_step_hist=True)

    # run adaptive solvers
    print("Adaptive ERK32 solver:")
    Y_E32, success = E32.Evolve(tvals, Y0)
    step_hist_E32 = E32.get_step_history()
    err_E32 = np.linalg.norm(Y_E32 - np.transpose(Ytrue), 1)
    print("  steps = %5i  fails = %2i, error = %.2e\n" %
      (E32.get_num_steps(), E32.get_num_error_failures(), err_E32))

    print("Adaptive DIRK32 solver:")
    Y_D32, success = D32.Evolve(tvals, Y0)
    step_hist_D32 = D32.get_step_history()
    err_D32 = np.linalg.norm(Y_D32 - np.transpose(Ytrue), 1)
    print("  steps = %5i  fails = %2i, error = %.2e\n" %
      (D32.get_num_steps(), D32.get_num_error_failures(), err_D32))
    solver.reset()

    # create plots for adaptive runs
    plt.figure()
    plt.plot(step_hist_E32['t'], step_hist_E32['h'], 'r-', label='ERK32')
    plt.plot(step_hist_D32['t'], step_hist_D32['h'], 'b-', label='DIRK32')
    for i in range(len(step_hist_E32['t'])):
        if (step_hist_E32['err'][i] > 1.0):
            plt.plot(step_hist_E32['t'][i], step_hist_E32['h'][i], 'rx')
    for i in range(len(step_hist_D32['t'])):
        if (step_hist_D32['err'][i] > 1.0):
            plt.plot(step_hist_D32['t'][i], step_hist_D32['h'][i], 'bx')
    plt.xlabel('$t$')
    plt.ylabel('$h$')
    plt.title('Adaptive step history, G = %i' % (g))
    plt.legend()
    fname = 'adaptive_steps_G%i.png' % (g)
    plt.savefig(fname)

plt.show()

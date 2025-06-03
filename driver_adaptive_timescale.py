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
#   G = stiffness at slow time scale (-10)
#   w = variable time-scale separation factor (varies)
#
# This script uses adaptive explicit solvers to track the dynamical
# time scale as w is varied.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
import matplotlib.pyplot as plt
from AdaptERK import *

# KPR problem parameters
Tf = 5
Nt = 50
tvals = np.linspace(0, Tf, Nt+1)
e = 0.5
W = [1, 10, 100, 1000, 10000]  # time scale values to test
G = -10

# loop over time scale values, accumulating time step history plot
plt.figure()
for w in W:
    print("\nKPR problem with w = %i\n" % (w))

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

    # KPR right-hand side function
    def f(t, y):
        u = y[0]
        v = y[1]
        return (np.array([[G, e], [e, -1]])
                @ np.array([(-1 + u**2 - r(t)) / (2 * u),
                            (-2 + v**2 - s(t)) / (2 * v)])
                + np.array([rdot(t) / (2 * u), sdot(t) / (2 * np.sqrt(2 + s(t)))]))

    # tolerances
    rtol = 1.e-4
    atol = 1.e-11

    # create adaptive ERK and DIRK solvers
    E32 = AdaptERK(f, Y0, ERK32(), rtol=rtol, atol=atol, save_step_hist=True)

    # run adaptive solver
    Y_E32, success = E32.Evolve(tvals, Y0)
    step_hist_E32 = E32.get_step_history()
    err_E32 = np.linalg.norm(Y_E32 - np.transpose(Ytrue), 1)
    print("  steps = %5i  fails = %2i, error = %.2e\n" %
      (E32.get_num_steps(), E32.get_num_error_failures(), err_E32))

    # add step history to plot
    ltext = 'w = %i (mean = %.0e)' % (w, np.mean(step_hist_E32['h']))
    plt.plot(step_hist_E32['t'], step_hist_E32['h'], label=ltext)

plt.xlabel('$t$')
plt.ylabel('$h$')
plt.title('Adaptive step histories')
plt.legend()
fname = 'adaptive_steps_w.png'
plt.savefig(fname)
plt.show()

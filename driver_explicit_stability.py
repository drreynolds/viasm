#!/usr/bin/env python3
#
# Script to test the forward Euler and some fixed-step ERK methods on the
# Dahlquist test problem
#     y' = lambda*y, t in [0,0.5],
#     y(0) = 1,
# for lambda = -100, h in {0.005, 0.01, 0.02, 0.04}
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
from ForwardEuler import *
from ERK import *

# problem time interval and Dahlquist parameter
t0 = 0.0
tf = 0.4
lam = -100.0

# problem-defining functions
def ytrue(t):
    """
    Generates a numpy array containing the true solution to the IVP at a given input t.
    """
    return np.array([np.exp(lam*t)])
def f(t,y):
    """
    Right-hand side function, f(t,y), for the Dahlquist IVP
    """
    return (lam*y)

# shared testing data
Nout = 11    # includes initial condition
tspan = np.linspace(t0, tf, Nout)
hvals = np.array([0.005, 0.01, 0.02, 0.04])

# create true solution results
Ytrue = np.zeros((Nout,1))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])

# Forward Euler: loop over time step sizes; call stepper and compute errors
FE = ForwardEuler(f)
print("\nForward Euler:")
for h in hvals:

    # set initial condition and call stepper
    y0 = Ytrue[0,:]
    print("  h = ", h, ":")
    FE.reset()
    Y, success = FE.Evolve(tspan, y0, h)

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Ytrue)
    for i in range(Nout):
        print("    y(%.1f) = %9.6f   |error| = %.2e" % (tspan[i], Y[i,0], Yerr[i,0]))
    print("  overall:  steps = %5i  abserr = %9.2e\n" % (FE.get_num_steps(), np.linalg.norm(Yerr,np.inf)))

# RK4: loop over time step sizes; call stepper and compute errors
RK4 = ERK(f, ERK4())
print("\n4th order explicit Runge-Kutta:")
for h in hvals:

    # set initial condition and call stepper
    y0 = Ytrue[0,:]
    print("  h = ", h, ":")
    RK4.reset()
    Y, success = RK4.Evolve(tspan, y0, h)

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Ytrue)
    for i in range(Nout):
        print("    y(%.1f) = %9.6f   |error| = %.2e" % (tspan[i], Y[i,0], Yerr[i,0]))
    print("  overall:  steps = %5i  abserr = %9.2e\n" % (RK4.get_num_steps(), np.linalg.norm(Yerr,np.inf)))

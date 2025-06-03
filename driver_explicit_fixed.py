#!/usr/bin/env python3
#
# Script to test explicit fixed-step Runge--Kutta methods.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
from ERK import *

# problem time interval
t0 = 0.0
tf = 1.0

# problem-definining functions and initial conditions
def f(t,y):
    """ ODE RHS function """
    return -y*np.exp(-t)
def f_t(t,y):
    """ t-derivative of ODE RHS function """
    return y*np.exp(-t)
def f_y(t,y):
    """ y-derivative of ODE RHS function """
    return np.array([[-np.exp(-t)]])
def ytrue(t):
    """ Analytical solution """
    return np.exp(np.exp(-t)-1.0)

# shared testing data
Nout = 3   # includes initial condition
tspan = np.linspace(t0, tf, Nout)

# create true solution results
Ytrue = np.zeros((Nout,1))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])

# time steps to try
hvals = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005])
errs = np.zeros(hvals.size)

#### Heun ####
print("\nHeun:")
H = ERK(f, Heun())
for idx, h in enumerate(hvals):

    # set initial condition and call stepper
    y0 = Ytrue[0,:]
    print("  h = ", h, ":")
    H.reset()
    Y, success = H.Evolve(tspan, y0, h)

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Ytrue)
    errs[idx] = np.linalg.norm(Yerr,np.inf)
    for i in range(Nout):
        print("    y(%.1f) = %9.6f   |error| = %.2e" % (tspan[i], Y[i,0], Yerr[i,0]))
    print("  overall:  steps = %5i  nrhs = %5i  abserr = %9.2e  relerr = %9.2e\n" %
          (H.get_num_steps(), H.get_num_rhs(), errs[idx], np.linalg.norm(Yerr/Ytrue,np.inf)))
orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))

#### ERK2 ####
print("\nERK2:")
E2 = ERK(f, ERK2())
for idx, h in enumerate(hvals):

    # set initial condition and call stepper
    y0 = Ytrue[0,:]
    print("  h = ", h, ":")
    E2.reset()
    Y, success = E2.Evolve(tspan, y0, h)

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Ytrue)
    errs[idx] = np.linalg.norm(Yerr,np.inf)
    for i in range(Nout):
        print("    y(%.1f) = %9.6f   |error| = %.2e" % (tspan[i], Y[i,0], Yerr[i,0]))
    print("  overall:  steps = %5i  nrhs = %5i  abserr = %9.2e  relerr = %9.2e\n" %
          (E2.get_num_steps(), E2.get_num_rhs(), errs[idx], np.linalg.norm(Yerr/Ytrue,np.inf)))
orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))

#### ERK3 ####
print("\nERK3:")
E3 = ERK(f, ERK3())
for idx, h in enumerate(hvals):

    # set initial condition and call stepper
    y0 = Ytrue[0,:]
    print("  h = ", h, ":")
    E3.reset()
    Y, success = E3.Evolve(tspan, y0, h)

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Ytrue)
    errs[idx] = np.linalg.norm(Yerr,np.inf)
    for i in range(Nout):
        print("    y(%.1f) = %9.6f   |error| = %.2e" % (tspan[i], Y[i,0], Yerr[i,0]))
    print("  overall:  steps = %5i  nrhs = %5i  abserr = %9.2e  relerr = %9.2e\n" %
          (E3.get_num_steps(), E3.get_num_rhs(), errs[idx], np.linalg.norm(Yerr/Ytrue,np.inf)))
orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))

#### ERK4 ####
print("\nERK4:")
E4 = ERK(f, ERK4())
for idx, h in enumerate(hvals):

    # set initial condition and call stepper
    y0 = Ytrue[0,:]
    print("  h = ", h, ":")
    E4.reset()
    Y, success = E4.Evolve(tspan, y0, h)

    # output solution, errors, and overall error
    Yerr = np.abs(Y-Ytrue)
    errs[idx] = np.linalg.norm(Yerr,np.inf)
    for i in range(Nout):
        print("    y(%.1f) = %9.6f   |error| = %.2e" % (tspan[i], Y[i,0], Yerr[i,0]))
    print("  overall:  steps = %5i  nrhs = %5i  abserr = %9.2e  relerr = %9.2e\n" %
          (E4.get_num_steps(), E4.get_num_rhs(), errs[idx], np.linalg.norm(Yerr/Ytrue,np.inf)))
orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
print('estimated order: max = ', np.max(orders), ',  avg = ', np.average(orders))

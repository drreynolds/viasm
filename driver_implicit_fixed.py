#!/usr/bin/env python3
#
# Script to test various fixed-step implicit methods on the
# scalar-valued ODE problem
#    y' = lambda*y + (1-lambda)*cos(t) - (1+lambda)*sin(t), t in [0,5],
#    y(0) = 1.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
from ImplicitSolver import *
from BackwardEuler import *
from DIRK import *

# problem time interval and parameters
t0 = 0.0
tf = 5.0

# problem-defining functions
def ytrue(t):
    """ Generates a numpy array containing the true solution to the IVP at a given input t. """
    return np.array( [np.sin(t) + np.cos(t)] )
def f(t,y,lam):
    """ Right-hand side function, f(t,y), for the IVP """
    return np.array( [lam*y[0] + (1.0-lam)*np.cos(t) - (1.0+lam)*np.sin(t)] )
def J(t,y,lam):
    """ Jacobian (in dense matrix format) of the right-hand side function, J(t,y) = df/dy """
    return np.array( [ [lam] ] )

# construct implicit solver
solver = ImplicitSolver(J, solver_type='dense', maxiter=20, rtol=1e-9, atol=1e-12, Jfreq=2)

# shared testing data
Nout = 6   # includes initial condition
tspan = np.linspace(t0, tf, Nout)
Ytrue = np.zeros((Nout, 1))
for i in range(Nout):
    Ytrue[i,:] = ytrue(tspan[i])
y0 = ytrue(t0)
lambdas = np.array( (-1.0, -10.0, -50.0) )
hvals = 1.0 / np.linspace(1, 7, 7)
errs = np.zeros(hvals.size)

# test runner function
def RunTest(stepper, name):

    print("\n", name, " tests:", sep='')
    # loop over stiffness values
    for lam in lambdas:

        print("  lambda = " , lam, ":", sep='')
        for idx, h in enumerate(hvals):
            print("    h = %.3f:" % (h), sep='', end='')
            stepper.reset()
            stepper.sol.reset()
            # Note that this is where we provide the rhs function parameter lam -- the "," is
            # required to ensure that args is an iterable (and not a float).
            Y, success = stepper.Evolve(tspan, y0, h, args=(lam,))
            Yerr = np.abs(Y-Ytrue)
            errs[idx] = np.linalg.norm(Yerr,np.inf)
            if (success):
                print("  solves = %4i  Niters = %6i  NJevals = %5i  abserr = %8.2e" %
                      (stepper.get_num_solves(), stepper.sol.get_total_iters(),
                       stepper.sol.get_total_setups(), errs[idx]))
        orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
        print('    estimated order:  max = %.2f,  avg = %.2f' %
              (np.max(orders), np.average(orders)))


# Backward Euler tests
BE = BackwardEuler(f, solver)
RunTest(BE, 'Backward Euler')

# Alexander3 tests
Alex3 = DIRK(f, solver, Alexander3())
RunTest(Alex3, 'Alexander-3')

# Crouzeix & Raviart tests
CR3 = DIRK(f, solver, CrouzeixRaviart3())
RunTest(CR3, 'Crouzeix & Raviart-3')

# SDIRK5 tests
SD5 = DIRK(f, solver, SDIRK5())
RunTest(SD5, 'SDIRK5')

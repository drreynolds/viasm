#!/usr/bin/env python3
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC
"""
Script to test the ImplicitSolver class for a "steady" problem.
"""
import numpy as np
import sys
from ImplicitSolver import *
from scipy.sparse import csc_matrix

# general parameters for all tests
maxit = 20
atols = np.array([1e-8, 1e-10, 1e-12])
rtols = np.array([1e-2, 1e-4, 1e-6])
ntols = len(rtols)
lags = [1,2,3]

# set the intial guess for all tests
x0 = np.array([0.95, 0.0, 0.01])

# set the residual function and various flavors of the Jacobian
def F(x):
    """ Residual function """
    return np.array([x[0] + 0.004*x[0] - 1e3*x[1]*x[2] - 1.0,
                     x[1] - 0.004*x[0] + 1e3*x[1]*x[2] + 30.0*x[1]*x[1],
                     x[2] - 30.0*x[1]*x[1]])
def J_dense(x):         # dense Jacobian
    """ Jacobian (in dense matrix format) of the residual function """
    return np.array([[1.004, -1e3*x[2], -1e3*x[1]],
                     [-0.004, 1.0 + 1e3*x[2] + 60.0*x[1], 1e3*x[1]],
                     [0.0, -60.0*x[1], 1.0]])
def J_sparse(x):        # sparse Jacobian
    """ Jacobian (in sparse matrix format) of the residual function """
    return csc_matrix(J_dense(x))

# set up ImplicitSolver objects for each Jacobian type:
# note: initial tolerances and Jacobian update frequency will be over-written during tests
# note2: we'll manually set up the linear solvers, but this step is typically taken care of
#        by the time integration method itself.
dense_solver  = ImplicitSolver(J_dense,  solver_type='dense',  maxiter=maxit, rtol=1e-2, atol=1e-8, Jfreq=1, steady=True)
sparse_solver = ImplicitSolver(J_sparse, solver_type='sparse', maxiter=maxit, rtol=1e-2, atol=1e-8, Jfreq=1, steady=True)

# call solvers for each tolerance, and in the case of direct linear solvers,
# for each modified Newton "lag" value
for i in range(ntols):
    print("\nCalling solvers with tolerances: rtol =", rtols[i], ", atol = ", atols[i])
    dense_solver.rtol  = rtols[i]
    dense_solver.atol  = atols[i]
    sparse_solver.rtol = rtols[i]
    sparse_solver.atol = atols[i]

    for l in lags:
        dense_solver.Jfreq = l
        dense_solver.reset()
        xsol, iters, success = dense_solver.solve(F, x0)
        print("  dense,  Jfreq=", l,": success =", success, ", iters =", iters)

        sparse_solver.Jfreq = l
        sparse_solver.reset()
        xsol, iters, success = sparse_solver.solve(F, x0)
        print("  sparse, Jfreq=", l,": success =", success, ", iters =", iters)

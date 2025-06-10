#!/usr/bin/env python3
#
# Script that runs various adaptive implicit methods on the Oregonator problem.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
import matplotlib.pyplot as plt
from ImplicitSolver import *
from ERK import *
from DIRK import *
from AdaptERK import *
from AdaptDIRK import *
import ReactionDiffusion as rd

# flags to enable/disable specific tests
runERK = True
runDIRK = True
runAdaptERK = True
runAdaptDIRK = True

# shared testing data
Nout = 20
tspan = np.linspace(rd.t0, rd.tf, Nout+1)
yref = np.zeros((Nout+1, rd.Nx-2), dtype=float)
for iout in range(Nout+1):
    yref[iout,:] = rd.utrue(rd.xgrid, tspan[iout])
solver = ImplicitSolver(rd.J, solver_type='sparse', maxiter=20,
                        rtol=1e-9, atol=1e-12, Jfreq=3)

if runERK:
    # fixed-step ERK
    ERK_4 = ERK(rd.f, ERK4())
    hvals = np.array([1e-5, 5e-6, 2.5e-6], dtype=float)
    errs = np.zeros(hvals.size) 
    print("\nExplicit RK4 solver:")
    for idx, h in enumerate(hvals):
        print("  h = ", h, ":")
        ERK_4.reset()
        Y_ERK, success = ERK_4.Evolve(tspan, yref[0,:], h)
        errs[idx] = np.linalg.norm(Y_ERK - yref, 1)
        print("  steps = %5i  nrhs = %5i, error = %.2e\n" % 
            (ERK_4.get_num_steps(), ERK_4.get_num_rhs(), errs[idx]))
    orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
    print('estimated order: ', np.median(orders))

if runDIRK:
    # fixed-step DIRK
    SDIRK_CR3 = DIRK(rd.f, solver, CrouzeixRaviart3())
    hvals = np.array([1e-2, 1e-3, 1e-4], dtype=float)
    errs = np.zeros(hvals.size) 
    print("\nDiagonally-implicit Crouzeix-Raviart-3 solver:")
    for idx, h in enumerate(hvals):
        print("  h = ", h, ":")
        SDIRK_CR3.reset()
        solver.reset()
        Y_DIRK, success = SDIRK_CR3.Evolve(tspan, yref[0,:], h)
        errs[idx] = np.linalg.norm(Y_DIRK - yref, 1)
        print("  steps = %5i  nsolves = %5i, error = %.2e\n" % 
            (SDIRK_CR3.get_num_steps(), SDIRK_CR3.get_num_solves(), errs[idx]))
    orders = np.log(errs[0:-2]/errs[1:-1])/np.log(hvals[0:-2]/hvals[1:-1])
    print('estimated order: ', np.median(orders))

if runAdaptERK:
    # adaptive-step ERK
    rtol = 1.e-6
    atol = 1.e-12
    print("\nAdaptive Dormand-Prince solver:")
    DP = AdaptERK(rd.f, yref[0,:], DormandPrince(), rtol=rtol, atol=atol)
    Y_DP, success = DP.Evolve(tspan, yref[0,:])
    err_DP = np.linalg.norm(Y_DP - yref, 1)
    print("  steps = %5i  fails = %2i, error = %.2e\n" % 
        (DP.get_num_steps(), DP.get_num_error_failures(), err_DP))

if runAdaptDIRK:
    # adaptive-step DIRK
    AD43 = AdaptDIRK(rd.f, yref[0,:], solver, ESDIRK43(), rtol=rtol, atol=atol)
    print("\nAdaptive DIRK43 solver:")
    Y_AD43, success = AD43.Evolve(tspan, yref[0,:])
    err_AD43 = np.linalg.norm(Y_AD43 - yref, 1)
    print("  steps = %5i  fails = %2i, error = %.2e\n" %
        (AD43.get_num_steps(), AD43.get_num_error_failures(), err_AD43))
    solver.reset()

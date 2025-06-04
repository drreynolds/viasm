#!/usr/bin/env python3
#
# Script that runs various Lie-Trotter subcycling methods and MRI methods
# on the nonlinear Kvaerno Prothero and Robinson problem:
#    [u]' = [ G  e ] [(-1+u^2-r)/(2u)] + [      r'(t)/(2u)        ]
#    [v]    [ e -1 ] [(-2+v^2-s)/(2v)]   [ s'(t)/(2*sqrt(2+s(t))) ]
#         = [fs(t,y)]
#           [ff(t,y)]
# where r(t) = 0.5*cos(t),  s(t) = cos(w*t),  0 < t < 5.
# This problem has analytical solution given by
#    u(t) = sqrt(1+r(t)),  v(t) = sqrt(2+s(t)).
#
# We use the parameters:
#   e = inter-variable coupling strength (0.5)
#   G = stiffness at slow time scale (-10)
#   w = variable time-scale separation factor (100)
#
# This script uses Lie-Trotter subcycling methods with components at various
# orders of accuracy to see if/how that affects accuracy.  It also runs MRI of
# various orders of accuracy to see how those compare.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
import matplotlib.pyplot as plt
from ERK import *
from LieTrotterSubcycling import *
from MRI import *

# methods to test
runLT = True
runMRI = True

# KPR problem parameters
Tf = 5
Nt = 25
tvals = np.linspace(0, Tf, Nt+1)
e = 0.5
w = 100
G = -10

# slow step sizes to try
Hvals = np.array([0.05, 0.025, 0.01, 0.005])
errs = np.zeros(Hvals.size)

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
Ytrue = np.zeros((Nt+1,2))
for i in range(Nt+1):
    Ytrue[i,:] = ytrue(tvals[i])

# KPR right-hand side functions
def fs(t, y):
    u = y[0]
    v = y[1]
    return (np.array([[G, e], [0, 0]])
            @ np.array([(-1 + u**2 - r(t)) / (2 * u),
                        (-2 + v**2 - s(t)) / (2 * v)])
            + np.array([rdot(t) / (2 * u), 0]))
def ff(t, y):
    u = y[0]
    v = y[1]
    return (np.array([[0, 0], [e, -1]])
            @ np.array([(-1 + u**2 - r(t)) / (2 * u),
                        (-2 + v**2 - s(t)) / (2 * v)])
            + np.array([0, sdot(t) / (2 * np.sqrt(2 + s(t)))]))

if (runLT):
    # Lie-Trotter-1
    print("\nLie-Trotter-Subcycling-1:")
    for idx, H in enumerate(Hvals):

        # set fast time step size
        h = H/w

        # create fast and slow steppers
        E1 = ERK(ff, ERK1(), h)
        LT1 = LieTrotterSubcycling(fs, ERK1(), E1, H)

        # set initial condition and call stepper
        y0 = Ytrue[0,:]
        print("  H = %f, h = %f:" % (H, h))
        Y, success = LT1.Evolve(tvals, y0)

        # output solution, errors, and overall error
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        print("   steps (s,f) = (%i, %i)  nrhs (s,f) = (%i, %i)  err = %.1e" %
            (LT1.get_num_steps(), E1.get_num_steps(), LT1.get_num_rhs(), E1.get_num_rhs(), errs[idx]))
    orders = np.log(errs[0:-2]/errs[1:-1])/np.log(Hvals[0:-2]/Hvals[1:-1])
    print('estimated order: ', np.median(orders))


    # Lie-Trotter-2
    print("\nLie-Trotter-Subcycling-2:")
    for idx, H in enumerate(Hvals):

        # set fast time step size
        h = H/w

        # create fast and slow steppers
        E2 = ERK(ff, ERK2(), h)
        LT2 = LieTrotterSubcycling(fs, ERK2(), E2, H)

        # set initial condition and call stepper
        y0 = Ytrue[0,:]
        print("  H = %f, h = %f:" % (H, h))
        Y, success = LT2.Evolve(tvals, y0)

        # output solution, errors, and overall error
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        print("   steps (s,f) = (%i, %i)  nrhs (s,f) = (%i, %i)  err = %.1e" %
            (LT2.get_num_steps(), E2.get_num_steps(), LT2.get_num_rhs(), E2.get_num_rhs(), errs[idx]))
    orders = np.log(errs[0:-2]/errs[1:-1])/np.log(Hvals[0:-2]/Hvals[1:-1])
    print('estimated order: ', np.median(orders))


    # Lie-Trotter-3
    print("\nLie-Trotter-Subcycling-3:")
    for idx, H in enumerate(Hvals):

        # set fast time step size
        h = H/w

        # create fast and slow steppers
        E3 = ERK(ff, ERK3(), h)
        LT3 = LieTrotterSubcycling(fs, ERK3(), E3, H)

        # set initial condition and call stepper
        y0 = Ytrue[0,:]
        print("  H = %f, h = %f:" % (H, h))
        Y, success = LT3.Evolve(tvals, y0)

        # output solution, errors, and overall error
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        print("   steps (s,f) = (%i, %i)  nrhs (s,f) = (%i, %i)  err = %.1e" %
            (LT3.get_num_steps(), E3.get_num_steps(), LT3.get_num_rhs(), E3.get_num_rhs(), errs[idx]))
    orders = np.log(errs[0:-2]/errs[1:-1])/np.log(Hvals[0:-2]/Hvals[1:-1])
    print('estimated order: ', np.median(orders))


if (runMRI):
    # MRI-GARK-ERK22a
    print("\nMRI-GARK-ERK22a:")
    for idx, H in enumerate(Hvals):

        # set fast time step size
        h = H/w

        # set initial condition
        y0 = Ytrue[0,:]

        # create fast and slow steppers
        E2 = ERK(ff, ERK2(), h)
        MRI2 = MRI(y0, fs, ff, MRIGARKERK22a(), E2, H)

        # call stepper
        print("  H = %f, h = %f:" % (H, h))
        Y, success = MRI2.Evolve(tvals, y0)

        # output solution, errors, and overall error
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        print("   steps (s,f) = (%i, %i)  nrhs (s,f) = (%i, %i)  err = %.1e" %
            (MRI2.get_num_steps(), E2.get_num_steps(), MRI2.get_num_rhs(), E2.get_num_rhs(), errs[idx]))
    orders = np.log(errs[0:-2]/errs[1:-1])/np.log(Hvals[0:-2]/Hvals[1:-1])
    print('estimated order: ', np.median(orders))


    # MRI-GARK-ERK33a
    print("\nMRI-GARK-ERK33a:")
    for idx, H in enumerate(Hvals):

        # set fast time step size
        h = H/w

        # set initial condition
        y0 = Ytrue[0,:]

        # create fast and slow steppers
        E3 = ERK(ff, ERK3(), h)
        MRI3 = MRI(y0, fs, ff, MRIGARKERK33a(), E3, H)

        # call stepper
        print("  H = %f, h = %f:" % (H, h))
        Y, success = MRI3.Evolve(tvals, y0)

        # output solution, errors, and overall error
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        print("   steps (s,f) = (%i, %i)  nrhs (s,f) = (%i, %i)  err = %.1e" %
            (MRI3.get_num_steps(), E3.get_num_steps(), MRI3.get_num_rhs(), E3.get_num_rhs(), errs[idx]))
    orders = np.log(errs[0:-2]/errs[1:-1])/np.log(Hvals[0:-2]/Hvals[1:-1])
    print('estimated order: ', np.median(orders))


    # MRI-GARK-ERK45a
    print("\nMRI-GARK-ERK45a:")
    for idx, H in enumerate(Hvals):

        # set fast time step size
        h = H/w

        # set initial condition
        y0 = Ytrue[0,:]

        # create fast and slow steppers
        E4 = ERK(ff, ERK4(), h)
        MRI4 = MRI(y0, fs, ff, MRIGARKERK45a(), E4, H)

        # call stepper
        print("  H = %f, h = %f:" % (H, h))
        Y, success = MRI4.Evolve(tvals, y0)

        # output solution, errors, and overall error
        Yerr = np.abs(Y-Ytrue)
        errs[idx] = np.linalg.norm(Yerr,np.inf)
        print("   steps (s,f) = (%i, %i)  nrhs (s,f) = (%i, %i)  err = %.1e" %
            (MRI4.get_num_steps(), E4.get_num_steps(), MRI4.get_num_rhs(), E4.get_num_rhs(), errs[idx]))
    orders = np.log(errs[0:-2]/errs[1:-1])/np.log(Hvals[0:-2]/Hvals[1:-1])
    print('estimated order: ', np.median(orders))

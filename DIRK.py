# DIRK.py
#
# Fixed-stepsize diagonally-implicit Runge--Kutta stepper class
# implementation file.
#
# Also contains functions to return specific DIRK Butcher tables.
#
# Class to perform fixed-stepsize time evolution of the IVP
#      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
# using a diagonally-implicit Runge--Kutta (DIRK) time stepping
# method.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
import sys
sys.path.append('..')
from shared.ImplicitSolver import *

class DIRK:
    """
    Fixed stepsize diagonally-implicit Runge--Kutta class

    The five required arguments when constructing a DIRK object are a
    function for the IVP right-hand side, an implicit solver to use,
    and a Butcher table:
        f = ODE RHS function with calling syntax f(t,y).
        sol = algebraic solver object to use [ImplicitSolver]
        A = Runge--Kutta stage coefficients (s*s matrix)
        b = Runge--Kutta solution weights (s array)
        c = Runge--Kutta abcissae (s array).
        h = (optional) input with stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, f, sol, A, b, c, h=0.0):
        # required inputs
        self.f = f
        self.sol = sol
        self.A = A
        self.b = b
        self.c = c

        # optional inputs
        self.h = h

        # internal data
        self.steps = 0
        self.nsol = 0
        self.s = c.size

        # check for legal table
        if ((np.size(c,0) != self.s) or (np.size(A,0) != self.s) or
            (np.size(A,1) != self.s) or (np.linalg.norm(A - np.tril(A,0), np.inf) > 1e-14)):
            raise ValueError("DIRK ERROR: incompatible Butcher table supplied")

    def dirk_step(self, t, y, args=()):
        """
        Usage: t, y, success = dirk_step(t, y, args)

        Utility routine to take a single diagonally-implicit RK time step,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # loop over stages, computing RHS vectors
        for i in range(self.s):

            # construct "data" for this stage solve
            self.data = np.copy(y)
            for j in range(i):
                self.data += self.h * self.A[i,j] * self.k[j,:]

            # construct implicit residual and Jacobian solver for this stage
            tstage = t + self.h*self.c[i]
            F = lambda zcur: zcur - self.data - self.h * self.A[i,i] * self.f(tstage, zcur, *args)
            self.sol.setup_linear_solver(tstage, -self.h * self.A[i,i], args)

            # perform implicit solve, and return on solver failure
            self.z, iters, success = self.sol.solve(F, y)
            self.nsol += 1
            if (not success):
                return t, y, False

            # store RHS at this stage
            self.k[i,:] = self.f(tstage, self.z, *args)

        # update time step solution
        for i in range(self.s):
            y += self.h * self.b[i] * self.k[i,:]
        t += self.h
        self.steps += 1
        return t, y, True

    def reset(self):
        """ Resets the accumulated number of steps """
        self.steps = 0
        self.nsol = 0

    def get_num_steps(self):
        """ Returns the accumulated number of steps """
        return self.steps

    def get_num_solves(self):
        """ Returns the accumulated number of implicit solves """
        return self.nsol

    def Evolve(self, tspan, y0, h=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h, args)

        The fixed-step DIRK evolution routine

        Inputs:  tspan holds the current time interval, [t0, tf], including any
                     intermediate times when the solution is desired, i.e.
                     [t0, t1, ..., tf]
                 y holds the initial condition, y(t0)
                 h optionally holds the requested step size (if it is not
                     provided then the stored value will be used)
                 args holds optional equation parameters used when evaluating
                     the RHS.
        Outputs: Y holds the computed solution at all tspan values,
                     [y(t0), y(t1), ..., y(tf)]
                 success = True if the solver traversed the interval,
                     false if an integration step failed [bool]
        """

        # set time step for evoluation based on input-vs-stored value
        if (h != 0.0):
            self.h = h

        # raise error if step size was never set
        if (self.h == 0.0):
            raise ValueError("ERROR: DIRK::Evolve called without specifying a nonzero step size")

        # verify that tspan values are separated by multiples of h
        for n in range(tspan.size-1):
            hn = tspan[n+1]-tspan[n]
            if (abs(round(hn/self.h) - (hn/self.h)) > 100*np.sqrt(np.finfo(h).eps)*abs(self.h)):
                raise ValueError("input values in tspan (%e,%e) are not separated by a multiple of h = %e" % (tspan[n],tspan[n+1],h))

        # initialize output, and set first entry corresponding to initial condition
        y = y0.copy()
        Y = np.zeros((tspan.size,y0.size))
        Y[0,:] = y

        # initialize internal solution-vector-sized data
        self.k = np.zeros((self.s, y0.size), dtype=float)
        self.z = y0.copy()
        self.data = y0.copy()

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required
            N = int(round((tspan[iout]-tspan[iout-1])/self.h))

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform diagonally-implicit Runge--Kutta update
                t, y, success = self.dirk_step(t, y, args)
                if (not success):
                    print("DIRK::Evolve error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True

def Alexander3():
    """
    Usage: A, b, c, p = Alexander3()

    Utility routine to return the DIRK table corresponding to
    Alexander's 3-stage O(h^3) method.

    Outputs: A holds the Runge--Kutta stage coefficients
             b holds the Runge--Kutta solution weights
             c holds the Runge--Kutta abcissae
             p holds the Runge--Kutta method order
    """
    alpha = 0.43586652150845906
    tau2 = 0.5*(1.0+alpha)
    A = np.array(((alpha, 0.0, 0.0),
                  (tau2-alpha, alpha, 0.0),
                  (-0.25*(6.0*alpha*alpha - 16.0*alpha + 1.0),
                   0.25*(6.0*alpha*alpha - 20*alpha + 5.0), alpha)))
    b = np.array((-0.25*(6.0*alpha*alpha - 16.0*alpha + 1.0),
                  0.25*(6.0*alpha*alpha - 20*alpha + 5.0), alpha))
    c = np.array((alpha, tau2, 1.0))
    p = 3
    return A, b, c, p

def CrouzeixRaviart3():
    """
    Usage: A, b, c, p = CrouzeixRaviart3()

    Utility routine to return the DIRK table corresponding to
    Crouzeix & Raviart's 3-stage O(h^4) method.

    Outputs: A holds the Runge--Kutta stage coefficients
             b holds the Runge--Kutta solution weights
             c holds the Runge--Kutta abcissae
             p holds the Runge--Kutta method order
    """
    gamma = 1.0/np.sqrt(3.0)*np.cos(np.pi/18.0) + 0.5
    delta = 1.0/(6.0*(2.0*gamma-1.0)*(2.0*gamma-1.0))
    A = np.array(((gamma, 0.0, 0.0),
                  (0.5-gamma, gamma, 0.0),
                  (2.0*gamma, 1.0-4.0*gamma, gamma)))
    b = np.array((delta, 1.0-2.0*delta, delta))
    c = np.array((gamma, 0.5, 1.0-gamma))
    p = 4
    return A, b, c, p

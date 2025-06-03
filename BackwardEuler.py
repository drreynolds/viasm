# BackwardEuler.py
#
# Fixed-stepsize backward Euler stepper class implementation file.
#
# Class to perform fixed-stepsize time evolution of the IVP
#      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
# using the backward Euler (implicit Euler) time stepping method.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
import sys
from ImplicitSolver import *

class BackwardEuler:
    """
    Fixed stepsize backward Euler class

    The two required arguments when constructing a BackwardEuler object
    are a function for the IVP right-hand side, and an implicit solver to use:
        f = ODE RHS function with calling syntax f(t,y).
        sol = algebraic solver object to use [ImplicitSolver]
        h = (optional) input with stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, f, sol, h=0.0):
        # required inputs
        self.f = f
        self.h = h
        self.sol = sol
        # internal data
        self.steps = 0

    def backward_euler_step(self, t, y, args=()):
        """
        Usage: t, y, success = backward_euler_step(t, y, args)

        Utility routine to take a single backward Euler time step,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # update t for this step
        t += self.h

        # create implicit residual and Jacobian solver for this step
        F = lambda ynew: ynew - y - self.h * self.f(t, ynew, *args)
        self.sol.setup_linear_solver(t, -self.h, args)

        # perform implicit solve, and return on solver failure
        y, iters, success = self.sol.solve(F, y)
        self.steps += 1
        return t, y, success

    def reset(self):
        """ Resets the accumulated number of steps """
        self.steps = 0

    def get_num_steps(self):
        """ Returns the accumulated number of steps """
        return self.steps

    def get_num_solves(self):
        """ Returns the accumulated number of implicit solves """
        return self.steps

    def Evolve(self, tspan, y0, h=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h, args)

        The fixed-step backward Euler evolution routine

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
        import numpy as np

        # set time step for evoluation based on input-vs-stored value
        if (h != 0.0):
            self.h = h

        # raise error if step size was never set
        if (self.h == 0.0):
            raise ValueError("ERROR: BackwardEuler::Evolve called without specifying a nonzero step size")

        # verify that tspan values are separated by multiples of h
        for n in range(tspan.size-1):
            hn = tspan[n+1]-tspan[n]
            if (abs(round(hn/self.h) - (hn/self.h)) > np.sqrt(np.finfo(h).eps)):
                raise ValueError("input values in tspan (%e,%e) are not separated by a multiple of h = %e" % (tspan[n],tspan[n+1],h))

        # initialize output, and set first entry corresponding to initial condition
        y = y0.copy()
        Y = np.zeros((tspan.size,y0.size))
        Y[0,:] = y

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required
            N = int(round((tspan[iout]-tspan[iout-1])/self.h))

            # reset "current" (t,y) that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform backward Euler step
                t, y, success = self.backward_euler_step(t, y, args)
                if (not success):
                    print("BackwardEuler::Evolve error in time step at t =", t)
                    return Y, False

            # store current result in output array
            Y[iout,:] = y.copy()

        # return with "success" flag set to True
        return Y, True

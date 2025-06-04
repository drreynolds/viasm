# LieTrotterSubcycling.py
#
# Fixed-stepsize Lie-Trotter based subcycling time stepper class implementation file.
#
# Class to perform fixed-stepsize, but subcycled, time evolution of the IVP
#      y' = fs(t,y) + ff(t,y),  t in [t0, Tf],  y(t0) = y0
# using a fixed step size ERK method for the sub-IVP
#      y' = fs(t,y),  y(tk) = yk,  t in [t_k, t_k + h]
# to compute ytmp at t_k+h, and any object that supports the "Evolve" routine
# for the sub-IVP
#      y' = ff(t,y),  y(tk) = ytmp,  t in [t_k, t_k + h].
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np

class LieTrotterSubcycling:
    """
    Fixed stepsize Lie-Trotter subcycling time stepper class

    The three required arguments when constructing a LieTrotterSubcycling object
    are a function for the "slow" IVP right-hand side, an explicit Runge--Kutta
    Butcher table to use for the slow IVP, and a solver for the "fast" sub-IVP:
        fs = ODE RHS function with calling syntax fs(t,y).
        B = Explicit Runge--Kutta Butcher table.
        FastSolver = object that implements the "Evolve" method for the fast IVP.
        h = (optional) input with requested stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, fs, B, FastSolver, H=0.0):
        # required inputs
        self.fs = fs
        self.A = B['A']
        self.b = B['b']
        self.c = B['c']
        if not hasattr(FastSolver, 'Evolve'):
            raise ValueError("FastSolver must implement the Evolve method")
        self.FastSolver = FastSolver

        # optional inputs
        self.H = H

        # internal data
        self.steps = 0
        self.nrhs = 0
        self.s = self.c.size

        # check for legal table
        if ((np.size(self.b,0) != self.s) or (np.size(self.A,0) != self.s) or
            (np.size(self.A,1) != self.s) or (np.linalg.norm(self.A - np.tril(self.A,-1), np.inf) > 1e-14)):
            raise ValueError("LieTrotterSubcycling ERROR: incompatible Butcher table supplied")

    def step(self, t, y, H, args=()):
        """
        Usage: t, y, success = step(t, y, H, args)

        Utility routine to take a single Lie-Trotter subcycled time step of size H,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # loop over slow stages, computing RHS vectors
        self.k[0,:] = self.fs(t, y, *args)
        self.nrhs += 1
        for i in range(1,self.s):
            self.z = np.copy(y)
            for j in range(i):
                self.z += H * self.A[i,j] * self.k[j,:]
            self.k[i,:] = self.fs(t + self.c[i] * H, self.z, *args)
            self.nrhs += 1

        # compute intermediate time step solution
        for i in range(self.s):
            y += H * self.b[i] * self.k[i,:]

        # call fast solver to evolve the sub-IVP
        tspan = np.array([t, t + H])
        ytmp, success = self.FastSolver.Evolve(tspan, y, args=args)

        # update current solution, time, and step counter, and return
        y = ytmp[1,:]  # extract the solution at t + h
        t += H
        self.steps += 1
        return t, y, success

    def update_rhs(self, fs):
        """ Updates the slow RHS function (cannot change vector dimensions) """
        self.fs = fs

    def reset(self):
        """ Resets the accumulated number of steps """
        self.steps = 0
        if hasattr(self.FastSolver, 'reset'):
            self.FastSolver.reset()

    def get_num_steps(self):
        """ Returns the accumulated number of slow steps """
        return self.steps

    def get_num_rhs(self):
        """ Returns the accumulated number of RHS evaluations """
        return self.nrhs

    def Evolve(self, tspan, y0, H=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h, args)

        The fixed-step Lie Trotter subcycled evolution routine

        Inputs:  tspan holds the current time interval, [t0, tf], including any
                     intermediate times when the solution is desired, i.e.
                     [t0, t1, ..., tf]
                 y holds the initial condition, y(t0)
                 H optionally holds the requested MRI step size (if it is not
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
        if (H != 0.0):
            self.H = H

        # raise error if step size was never set
        if (self.H == 0.0):
            raise ValueError("ERROR: LieTrotterSubcycling::Evolve called without specifying a nonzero step size")

        # initialize output, and set first entry corresponding to initial condition
        y = y0.copy()
        Y = np.zeros((tspan.size,y0.size))
        Y[0,:] = y

        # initialize internal solution-vector-sized data
        self.k = np.zeros((self.s, y0.size), dtype=float)
        self.z = y0.copy()

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required, and the actual step size to use
            N = int(np.ceil((tspan[iout]-tspan[iout-1])/self.H))
            H = (tspan[iout]-tspan[iout-1]) / N

            # reset "current" (t,y) that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform forward Euler update
                t, y, success = self.step(t, y, H, args)
                if (not success):
                    print("LieTrotterSubcycling error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True

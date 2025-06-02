# ERK.py
#
# Fixed-stepsize explicit Runge--Kutta stepper class implementation file.
#
# Also contains functions to return specific explicit Butcher tables.
#
# Class to perform fixed-stepsize time evolution of the IVP
#      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
# using an explicit Runge--Kutta (ERK) time stepping method.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np

class ERK:
    """
    Fixed stepsize explicit Runge--Kutta class

    The four required arguments when constructing an ERK object are a
    function for the IVP right-hand side, and a Butcher table:
        f = ODE RHS function with calling syntax f(t,y).
        B = Explicit Runge--Kutta Butcher table.
        h = (optional) input with stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, f, B, h=0.0):
        # required inputs
        self.f = f
        self.A = B['A']
        self.b = B['b']
        self.c = B['c']

        # optional inputs
        self.h = h

        # internal data
        self.steps = 0
        self.nrhs = 0
        self.s = self.c.size

        # check for legal table
        if ((np.size(self.b,0) != self.s) or (np.size(self.A,0) != self.s) or
            (np.size(self.A,1) != self.s) or (np.linalg.norm(self.A - np.tril(self.A,-1), np.inf) > 1e-14)):
            raise ValueError("ERK ERROR: incompatible Butcher table supplied")

    def erk_step(self, t, y, args=()):
        """
        Usage: t, y, success = erk_step(t, y, args)

        Utility routine to take a single explicit RK time step,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # loop over stages, computing RHS vectors
        self.k[0,:] = self.f(t, y, *args)
        self.nrhs += 1
        for i in range(1,self.s):
            self.z = np.copy(y)
            for j in range(i):
                self.z += self.h * self.A[i,j] * self.k[j,:]
            self.k[i,:] = self.f(t + self.c[i] * self.h, self.z, *args)
            self.nrhs += 1

        # update time step solution and tcur
        for i in range(self.s):
            y += self.h * self.b[i] * self.k[i,:]
        t += self.h
        self.steps += 1
        return t, y, True

    def reset(self):
        """ Resets the accumulated number of steps """
        self.steps = 0
        self.nrhs = 0

    def get_num_steps(self):
        """ Returns the accumulated number of steps """
        return self.steps

    def get_num_rhs(self):
        """ Returns the accumulated number of RHS evaluations """
        return self.nrhs

    def Evolve(self, tspan, y0, h=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h, args)

        The fixed-step explicit Runge--Kutta evolution routine.

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
            raise ValueError("ERROR: ERK::Evolve called without specifying a nonzero step size")

        # verify that tspan values are separated by multiples of h
        for n in range(tspan.size-1):
            hn = tspan[n+1]-tspan[n]
            if (abs(round(hn/self.h) - (hn/self.h)) > 100*np.sqrt(np.finfo(h).eps)*abs(self.h)):
                raise ValueError("input values in tspan (%e,%e) are not separated by a multiple of h = %e" % (tspan[n],tspan[n+1],h))

        # initialize output, and set first entry corresponding to initial condition
        y = y0.copy()
        Y = np.zeros((tspan.size, y0.size))
        Y[0,:] = y

        # initialize internal solution-vector-sized data
        self.k = np.zeros((self.s, y0.size), dtype=float)
        self.z = y0.copy()

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required
            N = int(round((tspan[iout]-tspan[iout-1])/self.h))

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform explicit Runge--Kutta update
                t, y, success = self.erk_step(t, y, args)
                if (not success):
                    print("erk error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True

def Heun():
    """
    Usage: B = Heun()

    Utility routine to return the ERK table corresponding to Heun's method.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array(((0.0, 0.0), (1.0, 0.0)))
    b = np.array((0.5, 0.5))
    c = np.array((0.0, 1.0))
    p = 2
    B = {'A': A, 'b':b, 'c':c, 'p': p}
    return B

def ERK2():
    """
    Usage: B = ERK2()

    Utility routine to return the ERK table corresponding
    to the standard 2nd-order ERK method.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array(((0.0, 0.0), (0.5, 0.0)))
    b = np.array((0.0, 1.0))
    c = np.array((0.0, 0.5))
    p = 2
    B = {'A': A, 'b':b, 'c':c, 'p': p}
    return B

def ERK3():
    """
    Usage: B = ERK3()

    Utility routine to return the ERK table corresponding
    to the standard 3rd-order ERK method.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array(((0.0, 0.0, 0.0), (2.0/3.0, 0.0, 0.0), (0.0, 2.0/3.0, 0.0)))
    b = np.array((0.25, 3.0/8.0, 3.0/8.0))
    c = np.array((0.0, 2.0/3.0, 2.0/3.0))
    p = 3
    B = {'A': A, 'b':b, 'c':c, 'p': p}
    return B

def ERK4():
    """
    Usage: B = ERK4()

    Utility routine to return the ERK table corresponding
    to the standard 4th-order ERK method.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array(((0.0, 0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.0, 0.0),
                  (0.0, 0.5, 0.0, 0.0),
                  (0.0, 0.0, 1.0, 0.0)))
    b = np.array((1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0))
    c = np.array((0.0, 0.5, 0.5, 1.0))
    p = 4
    B = {'A': A, 'b':b, 'c':c, 'p': p}
    return B

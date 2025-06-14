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

    The two required arguments when constructing an ERK object are a
    function for the IVP right-hand side, and a Butcher table:
        f = ODE RHS function with calling syntax f(t,y).
        B = Explicit Runge--Kutta Butcher table.
        h = (optional) input with requested stepsize to use for time stepping.
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

    def erk_step(self, t, y, h, args=()):
        """
        Usage: t, y, success = erk_step(t, y, h, args)

        Utility routine to take a single explicit RK time step of size h,
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
                self.z += h * self.A[i,j] * self.k[j,:]
            self.k[i,:] = self.f(t + self.c[i] * h, self.z, *args)
            self.nrhs += 1

        # update time step solution and tcur
        for i in range(self.s):
            y += h * self.b[i] * self.k[i,:]
        t += h
        self.steps += 1
        return t, y, True

    def update_rhs(self, f):
        """ Updates the RHS function (cannot change vector dimensions) """
        self.f = f

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

        # initialize output, and set first entry corresponding to initial condition
        y = y0.copy()
        Y = np.zeros((tspan.size, y0.size))
        Y[0,:] = y

        # initialize internal solution-vector-sized data
        self.k = np.zeros((self.s, y0.size), dtype=float)
        self.z = y0.copy()

        # loop over desired output times
        for iout in range(1,tspan.size):

            # determine how many internal steps are required, and the actual step size to use
            N = int(np.ceil((tspan[iout]-tspan[iout-1])/self.h))
            h = (tspan[iout]-tspan[iout-1]) / N

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform explicit Runge--Kutta update
                t, y, success = self.erk_step(t, y, h, args)
                if (not success):
                    print("erk error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True

def ERK1():
    """
    Usage: B = ERK1()

    Utility routine to return the ERK table corresponding to forward Euler, posed as an ERK method.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array((((0.0,),)))
    b = np.array((1.0,))
    c = np.array((0.0,))
    p = 1
    B = {'A': A, 'b':b, 'c':c, 'p': p}
    return B

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

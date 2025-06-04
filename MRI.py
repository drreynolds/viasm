# MRI.py
#
# Fixed-stepsize multirate infinitesimal (MRI) time stepper class implementation file.
#
# Class to perform fixed-stepsize MRI time evolution of the IVP
#      y' = fs(t,y) + ff(t,y),  t in [t0, Tf],  y(t0) = y0
# using a fixed step size explicit MRI method for the sub-IVP
#      y' = fs(t,y),  y(tk) = yk,  t in [t_k, t_{k+1}],
# and any object that supports the "Evolve" routine
# for the sub-IVPs
#      y' = ff(t,y) + r(t),  y(tk) = y_k,  t in [t_k, t_{k+1}].
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np

class MRI:
    """
    Fixed stepsize MRI time stepper class

    The four required arguments when constructing a MRI object
    are a function for the "slow" IVP right-hand side, an explicit Runge--Kutta
    Butcher table to use for the slow IVP, and a solver for the "fast" sub-IVP:
        y = template for the solution vector.
        fs = slow ODE RHS function with calling syntax fs(t,y).
        ff = fast ODE RHS function with calling syntax fs(t,y).
        C = explicit MRI coupling table, with the following keys:
            'G' = 3D numpy array of coupling coefficients, with G[0,:,:] corresponding
                  to the matrix of constant tendencies, G[1,:,:] the matrix of linear
                  tendencies, etc.  Note: these must be strictly lower triangular.
            'c' = vector of slow stage abscissae.  Note: these must be sorted such that
                  c[0] < c[1] < ... < c[end-1]
        FastSolver = object that implements the "Evolve" method for the fast IVP,
            and that support the "update_rhs" method to update the fast RHS function.
            This should support fast evolution intervals with different width, i.e., if
            using fixed substeps then there should be no restriction that the fast substep
            divide the interval evenly.
        H = (optional) input with requested stepsize to use for MRI time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, y, fs, ff, C, FastSolver, H=0.0):
        # required inputs
        self.fs = fs
        self.ff = ff
        self.G = C['G']
        self.c = C['c']
        if not hasattr(FastSolver, 'Evolve'):
            raise ValueError("FastSolver must implement the Evolve method")
        if not hasattr(FastSolver, 'update_rhs'):
            raise ValueError("FastSolver must implement the update_rhs method")
        self.FastSolver = FastSolver

        # optional inputs
        self.H = H

        # internal data
        self.steps = 0
        self.nrhs = 0
        self.s = self.c.size
        self.m = y.size
        self.Fs = np.zeros((self.m, self.s), dtype=float)

        # check for legal table
        Gabssum = np.sum(np.abs(self.G), axis=0)
        if ((np.linalg.norm(Gabssum - np.tril(Gabssum,-1), np.inf) > 1e-14)):
            raise ValueError("MRI ERROR: incompatible coupling table supplied")
        self.deltac = np.diff(self.c)
        if (np.any(self.deltac <= 0.0)):
            raise ValueError("MRI ERROR: coupling abscissae must be strictly increasing")

    def set_fast_rhs(self, tn, stage, H):
        """
        Usage: set_fast_rhs(tn, stage, H)

        Utility routine to update the RHS function in the fast solver based on the
        time at the start of the current MRI step, and the current MRI stage index.
        """
        # number of MRI "Gamma" matrices
        ngammas = np.size(self.G, 0)

        # starting 'time' for current stage solve
        tprev = tn + self.c[stage-1] * H

        # utility routine to convert t to theta
        def theta(t):
            return (t - tprev) / self.deltac[stage-1]

        # utility routine to construct a vector with powers of (theta/H)
        def thetapow(t):
            return np.power(theta(t)/H, np.arange(ngammas))

        # set Gamma matrix for this fast RHS
        Gamma = np.zeros((self.s, ngammas), dtype=float)
        for col in range(ngammas):
            for row in range(stage):
                Gamma[row,col] = self.G[col,stage,row]/self.deltac[stage-1]

        # set fast RHS function for the current stage
        def ff(tau, v):
            return self.Fs @ (Gamma @ thetapow(tau)) + self.ff(tau, v)
        self.FastSolver.update_rhs(ff)

    def step(self, t, y, H, args=()):
        """
        Usage: t, y, success = step(t, y, H, args)

        Utility routine to take a single MRI time step of size H,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # clear out Fs, and store first slow RHS evaluation (since stage 0 is explicit)
        self.Fs = np.zeros((y.size, self.s), dtype=float)
        self.Fs[:,0] = self.fs(t, y, *args)
        self.nrhs += 1

        # loop over remaining slow stages
        for stage in range(1, self.s):

            # update fast solver with the current forcing function
            self.set_fast_rhs(t, stage, H)

            # set fast time interval
            tspan = np.array([t + self.c[stage-1] * H, t + self.c[stage] * H])

            # call fast solver to evolve the sub-IVP
            ytmp, success = self.FastSolver.Evolve(tspan, y, args=args)
            if not success:
                return t, y, False

            # update current solution and store slow RHS evaluation
            y = ytmp[-1,:]
            self.Fs[:,stage] = self.fs(tspan[1], y, *args)
            self.nrhs += 1

        # update current time and step counter, and return
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
        Usage: Y, success = Evolve(tspan, y0, H, args)

        The fixed-step MRI evolution routine

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

        # set time step for evoluation based on input-vs-stored value
        if (H != 0.0):
            self.H = H

        # raise error if step size was never set
        if (self.H == 0.0):
            raise ValueError("ERROR: MRI::Evolve called without specifying a nonzero step size")

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
                    print("MRI error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True


def MRIGARKERK22a():
    """
    Usage: C = MRIGARKERK33a()

    Returns a dictionary with the coupling coefficients and abscissae
    for the explicit MRI-GARK-ERK33a method.
    """
    C = {}
    c2 = 0.5
    C['G'] = np.array([[[0, 0, 0],
                        [c2, 0, 0],
                        [-(2*c2*c2-2*c2+1)/(2*c2), 1/(2*c2), 0]],], dtype=float)
    C['c'] = np.array([0, c2, 1], dtype=float)
    return C


def MRIGARKERK33a():
    """
    Usage: C = MRIGARKERK33a()

    Returns a dictionary with the coupling coefficients and abscissae
    for the explicit MRI-GARK-ERK33a method.
    """
    C = {}
    C['G'] = np.array([[[0, 0, 0, 0],
                        [1/3, 0, 0, 0],
                        [-1/3, 2/3, 0, 0],
                        [0, -2/3, 1, 0]],
                       [[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1/2, 0, -1/2, 0]]], dtype=float)
    C['c'] = np.array([0, 1/3, 2/3, 1], dtype=float)
    return C


def MRIGARKERK45a():
    """
    Usage: C = MRIGARKERK45a()

    Returns a dictionary with the coupling coefficients and abscissae
    for the explicit MRI-GARK-ERK45a method.
    """
    C = {}
    C['G'] = np.array([[[0, 0, 0, 0, 0, 0],
                        [1/5, 0, 0, 0, 0, 0],
                        [-53/16, 281/80, 0, 0, 0, 0],
                        [-36562993/71394880, 34903117/17848720, -88770499/71394880, 0, 0,0],
                        [-7631593/71394880, -166232021/35697440, 6068517/1519040, 8644289/8924360, 0, 0],
                        [277061/303808, -209323/1139280, -1360217/1139280, -148789/56964, 147889/45120, 0]],
                       [[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [503/80, -503/80, 0, 0, 0, 0],
                        [-1365537/35697440, 4963773/7139488, -1465833/2231090, 0, 0, 0],
                        [66974357/35697440, 21445367/7139488, -3, -8388609/4462180, 0, 0],
                        [-18227/7520, 2, 1, 5, -41933/7520, 0]]], dtype=float)
    C['c'] = np.array([0, 1/5, 2/5, 3/5, 4/5, 1], dtype=float)
    return C
# DIRK.py
#
# Fixed-stepsive diagonally-implicit Runge--Kutta stepper class
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
from ImplicitSolver import *

class DIRK:
    """
    Fixed stepsize diagonally-implicit Runge--Kutta class

    The three required arguments when constructing a DIRK object are a
    function for the IVP right-hand side, an implicit solver to use,
    and a Butcher table:
        f = ODE RHS function with calling syntax f(t,y).
        sol = algebraic solver object to use [ImplicitSolver]
        B = diagonally-implicit Runge--Kutta Butcher table.
        h = (optional) input with requested stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, f, sol, B, h=0.0):
        # required inputs
        self.f = f
        self.sol = sol
        self.A = B['A']
        self.b = B['b']
        self.c = B['c']

        # optional inputs
        self.h = h

        # internal data
        self.steps = 0
        self.nsol = 0
        self.s = self.c.size

        # check for legal table
        if ((np.size(self.c,0) != self.s) or (np.size(self.A,0) != self.s) or
            (np.size(self.A,1) != self.s) or (np.linalg.norm(self.A - np.tril(self.A,0), np.inf) > 1e-14)):
            raise ValueError("DIRK ERROR: incompatible Butcher table supplied")

    def dirk_step(self, t, y, h, args=()):
        """
        Usage: t, y, success = dirk_step(t, y, h, args)

        Utility routine to take a single diagonally-implicit RK time step of size h,
        where the inputs (t,y) are overwritten by the updated versions.
        args is used for optional parameters of the RHS.
        If success==True then the step succeeded; otherwise it failed.
        """

        # loop over stages, computing RHS vectors
        for i in range(self.s):

            # construct "data" for this stage solve
            self.data = np.copy(y)
            for j in range(i):
                self.data += h * self.A[i,j] * self.k[j,:]

            # construct implicit residual and Jacobian solver for this stage
            tstage = t + h*self.c[i]
            F = lambda zcur: zcur - self.data - h * self.A[i,i] * self.f(tstage, zcur, *args)
            self.sol.setup_linear_solver(tstage, -h * self.A[i,i], args)

            # perform implicit solve, and return on solver failure
            self.z, iters, success = self.sol.solve(F, y)
            self.nsol += 1
            if (not success):
                return t, y, False

            # store RHS at this stage
            self.k[i,:] = self.f(tstage, self.z, *args)

        # update time step solution
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

            # determine how many internal steps are required, and the actual step size to use
            N = int(np.ceil((tspan[iout]-tspan[iout-1])/self.h))
            h = (tspan[iout]-tspan[iout-1]) / N

            # reset "current" t that will be evolved internally
            t = tspan[iout-1]

            # iterate over internal time steps to reach next output
            for n in range(N):

                # perform diagonally-implicit Runge--Kutta update
                t, y, success = self.dirk_step(t, y, h, args)
                if (not success):
                    print("DIRK::Evolve error in time step at t =", t)
                    return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with "success" flag
        return Y, True

def Alexander3():
    """
    Usage: B = Alexander3()

    Utility routine to return the DIRK table corresponding to
    Alexander's 3-stage O(h^3) method.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
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
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def CrouzeixRaviart3():
    """
    Usage: B = CrouzeixRaviart3()

    Utility routine to return the DIRK table corresponding to
    Crouzeix & Raviart's 3-stage O(h^4) method.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    gamma = 1.0/np.sqrt(3.0)*np.cos(np.pi/18.0) + 0.5
    delta = 1.0/(6.0*(2.0*gamma-1.0)*(2.0*gamma-1.0))
    A = np.array(((gamma, 0.0, 0.0),
                  (0.5-gamma, gamma, 0.0),
                  (2.0*gamma, 1.0-4.0*gamma, gamma)))
    b = np.array((delta, 1.0-2.0*delta, delta))
    c = np.array((gamma, 0.5, 1.0-gamma))
    p = 4
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def SDIRK5():
    """
    Usage: B = SDIRK5()

    Utility routine to return the SDIRK table corresponding to
    a 5-stage, 5th-order method.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array(((4024571134387/14474071345096, 0, 0, 0, 0),
        (9365021263232/12572342979331, 4024571134387/14474071345096, 0, 0, 0),
        (2144716224527/9320917548702, -397905335951/4008788611757, 4024571134387/14474071345096, 0, 0),
        (-291541413000/6267936762551, 226761949132/4473940808273, -1282248297070/9697416712681, 4024571134387/14474071345096, 0),
        (-2481679516057/4626464057815, -197112422687/6604378783090, 3952887910906/9713059315593, 4906835613583/8134926921134, 4024571134387/14474071345096)))
    b = np.array((-2522702558582/12162329469185, 1018267903655/12907234417901, 4542392826351/13702606430957, 5001116467727/12224457745473, 1509636094297/3891594770934))
    c = np.array((4024571134387/14474071345096, 5555633399575/5431021154178,  5255299487392/12852514622453, 3/20, 10449500210709/14474071345096))
    p = 5
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

def EDIRK744():
    """
    Usage: B = EDIRK744()

    Utility routine to return the EDDIRK table corresponding to
    a 4th-order accurate method with semilinear order 4.

    Outputs: B['A'] holds the Runge--Kutta stage coefficients
             B['b'] holds the Runge--Kutta solution weights
             B['c'] holds the Runge--Kutta abcissae
             B['p'] holds the Runge--Kutta method order
    """
    A = np.array(((0, 0, 0, 0, 0, 0, 0),
                  (66719178356146069/104971894986575178, 66719178356146069/104971894986575178, 0, 0, 0, 0, 0),
                  (11574878994758291/117719113355115783, -1858197540898696/70529361366069153, 11617133062216757/43245479316548780, 0, 0, 0, 0),
                  (312078294212599530/40823424700776821, 155312269009595199/86710391005988198, -743789150637775609/113352218631221311, 98271968880200657/179019545289054999, 0, 0, 0),
                  (1246868775297421168/137070970121741807, 114921713922407255/52367417556902641, -205947502305419261/24454220481972685, 18936671640200689/104159855867653343, 47397311839212708/127463680130367391, 0, 0),
                  (-151740509096074388/196613682401464609, 254369392774793867/44087509892864172, -73864359103986538/65744654972066205, -37706375961306427/179802732674457709, 7953265906419399/38933344132172515, 48325866641079469/46020097947328612, 0),
                  (3312403043354842/33496693975407517, -7745264544994559/74509708869668763, 75463258779378077/134382831179297809, -11696764876217691/132584149662964151, 9114026243344448/106054923174086269, 55946924902076/75756035623695139, 34834932759942553/78271243704016222)), dtype=float)
    b = A[-1,:]
    c = np.sum(A, axis=1)
    p = 4
    B = {'A': A, 'b': b, 'c': c, 'p': p}
    return B

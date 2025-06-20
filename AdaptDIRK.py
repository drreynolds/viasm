# AdaptDIRK.py
#
# Adaptive-stepsive diagonally-implicit Runge--Kutta solver class
# implementation file.
#
# Also contains functions to return specific embedded DIRK Butcher tables.
#
# Class to perform adaptive stepsize time evolution of the IVP
#      y' = f(t,y),  t in [t0, Tf],  y(t0) = y0
# using an embedded diagonally-implicit Runge--Kutta (DIRK) time stepping
# method.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
from ImplicitSolver import *

class AdaptDIRK:
    """
    Adaptive diagonally-implicit Runge--Kutta class

    The four required arguments when constructing a DIRK object are a
    function for the IVP right-hand side, an implicit solver to use,
    and a Butcher table:
        f = ODE RHS function with calling syntax f(t,y).
        y = numpy array with m entries.
        sol = algebraic solver object to use [ImplicitSolver]
        B = diagonally-implicit embedded Butcher table.
        h = (optional) input with stepsize to use for time stepping.
            Note that this MUST be set either here or in the Evolve call.
    """
    def __init__(self, f, y, sol, B, rtol=1e-3, atol=1e-14, maxit=1e6, bias=1.0, growth=50.0, safety=0.85, hmin=10*np.finfo(float).eps, save_step_hist=False):
        # required inputs
        self.f = f
        self.sol = sol
        self.A = B['A']
        self.b = B['b']
        self.c = B['c']
        self.d = B['d']
        self.minpq = min(B['p'], B['q'])

        # optional inputs
        self.rtol = rtol
        self.atol = np.ones(y.size)*atol
        self.maxit = maxit
        self.bias = bias
        self.growth = growth
        self.safety = safety
        self.hmin = hmin

        # internal data
        self.w = np.ones(y.size)
        self.yerr = np.zeros(y.size)
        self.ONEMSM = 1.0 - np.sqrt(np.finfo(float).eps)
        self.ONEPSM = 1.0 + np.sqrt(np.finfo(float).eps)
        self.fails = 0
        self.steps = 0
        self.nsol = 0
        self.save_step_hist = save_step_hist
        self.step_hist = {'t': [], 'h': [], 'err': []}
        self.error_norm = 0.0
        self.h = 0.0
        self.z = np.zeros(y.size)
        self.yt = np.zeros(y.size)
        self.data = np.zeros(y.size)
        self.s = len(self.b)
        self.k = np.zeros((self.s, y.size))

        # check for legal table
        if ((np.size(self.c,0) != self.s) or (np.size(self.A,0) != self.s) or
            (np.size(self.A,1) != self.s) or (np.linalg.norm(self.b-self.d) < 1e-14) or
            (np.linalg.norm(self.A - np.tril(self.A,0), np.inf) > 1e-14)):
            raise ValueError("AdaptDIRK ERROR: incompatible Butcher table supplied")

    def error_weight(self, y, w):
        """
        Error weight vector utility routine
        """
        for i in range(y.size):
            w[i] = self.bias / (self.atol[i] + self.rtol * np.abs(y[i]))
        return w

    def step(self, t, y, args=()):
        """
        Usage: t, y, success = step(t, y, args)

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

        # compute error estimate (and norm), and return
        self.yerr *= 0.0
        for i in range(self.s):
            self.yerr += self.h * (self.b[i] - self.d[i])* self.k[i,:]
        self.error_norm = max(np.linalg.norm(self.yerr*self.w, np.inf), 1.e-8)
        return t, y, True

    def Evolve(self, tspan, y0, h=0.0, args=()):
        """
        Usage: Y, success = Evolve(tspan, y0, h, args)

        The adaptive DIRK time step evolution routine

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
        # store input step size
        self.h = h

        # store sizes
        m = len(y0)
        N = len(tspan)-1

        # initialize output
        y = y0.copy()
        Y = np.zeros((N+1, m))
        Y[0,:] = y

        # set current time value
        t = tspan[0]

        # check for legal time span
        for n in range(N):
            if (tspan[n+1] < tspan[n]):
                raise ValueError("AdaptERK::Evolve illegal tspan")

        # initialize error weight vector, and check for legal tolerances
        self.w = self.error_weight(y, self.w)

        # estimate initial step size if not provided by user
        if (self.h == 0.0):

            # get ||y'(t0)||
            fn = self.f(t, y, *args)

            # estimate initial h value via linearization, safety factor
            self.error_norm = max(np.linalg.norm(fn*self.w, np.inf), 1.e-8)
            self.h = max(self.hmin, self.safety / self.error_norm)

        # iterate over output times
        for iout in range(1,N+1):

            # loop over internal steps to reach desired output time
            while ((tspan[iout]-t) > np.sqrt(np.finfo(float).eps*tspan[iout])):

                # enforce maxit -- if we've exceeded attempts, return with failure
                if (self.steps + self.fails > self.maxit):
                    print("AdaptDIRK: reached maximum iterations, returning with failure")
                    return Y, False

                # bound internal time step to not exceed next output time
                self.h = min(self.h, tspan[iout]-t)

                # reset temporary solution to current solution, and take DIRK step
                self.yt = y.copy()
                t, self.yt, success = self.step(t, self.yt, args)
                if (not success):
                    print("AdaptDIRK::Evolve error in time step at t =", t)
                    return Y, False

                # estimate step size growth/reduction factor based on error estimate
                eta = self.safety * self.error_norm**(-1.0/(self.minpq+1))  # step size growth factor
                eta = min(eta, self.growth)                             # limit maximum growth

                # store step size in history if requested
                if (self.save_step_hist):
                    self.step_hist['t'].append(t)
                    self.step_hist['h'].append(self.h)
                    self.step_hist['err'].append(self.error_norm)

                # check error
                if (self.error_norm < self.ONEPSM):  # successful step

                    # update current time, solution, error weights, work counter, and upcoming stepsize
                    t += self.h
                    y = self.yt.copy()
                    self.w = self.error_weight(y, self.w)
                    self.steps += 1
                    self.h *= eta

                else:                                 # failed step
                    self.fails += 1

                    # adjust step size, enforcing minimum and returning with failure if needed
                    if (self.h > self.hmin):                              # failure, but reduction possible
                        self.h = max(self.h * eta, self.hmin)
                    else:                                                 # failed with no reduction possible
                        print("AdaptDIRK: error test failed at h=hmin, returning with failure")
                        return Y, False

            # store current results in output arrays
            Y[iout,:] = y.copy()

        # return with successful solution
        return Y, True

    def set_rtol(self, rtol=1e-3):
        """ Resets the relative tolerance """
        self.rtol = rtol

    def set_atol(self, atol=1e-14):
        """ Resets the scalar- or vector-valued absolute tolerance """
        self.atol = np.ones(self.atol.size)*atol

    def set_maxit(self, maxit=1e6):
        """ Resets the maximum allowed iterations """
        self.maxit = maxit

    def set_bias(self, bias=2.0):
        """ Resets the error bias factor """
        self.bias = bias

    def set_growth(self, growth=50.0):
        """ Resets the maximum stepsize growth factor """
        self.growth = growth

    def set_safety(self, safety=0.95):
        """ Resets the stepsize safety factor """
        self.safety = safety

    def set_hmin(self, hmin=10*np.finfo(float).eps):
        """ Resets the minimum step size """
        self.hmin = hmin

    def update_rhs(self, f):
        """ Updates the RHS function (cannot change vector dimensions) """
        self.f = f

    def get_error_weight(self):
        """ Returns the current error weight vector """
        return self.w

    def get_error_vector(self):
        """ Returns the current error vector """
        return self.yerr

    def get_error_norm(self):
        """ Returns the scaled error norm """
        return self.error_norm

    def get_num_error_failures(self):
        """ Returns the total number of error test failures """
        return self.fails

    def get_num_steps(self):
        """ Returns the accumulated number of steps """
        return self.steps

    def get_num_solves(self):
        """ Returns the accumulated number of implicit solves """
        return self.nsol

    def get_current_step(self):
        """ Returns the current internal step size """
        return self.h

    def get_step_history(self):
        """ Returns the current step size history """
        return self.step_hist

    def reset(self):
        """ Resets the accumulated number of steps """
        self.fails = 0
        self.error_norm = 0.0
        self.nsol = 0
        self.steps = 0
        self.step_hist = {'t': [], 'h': [], 'err': []}


# embedded DIRK Butcher table routines

def SDIRK21():
    """
    Usage: B = SDIRK21()

    Utility routine to return the SDIRK table corresponding to
    an embedded method with order 2 and embedding order 1.

    Outputs: B['A'] holds the stage coefficients
             B['b'] holds the solution weights
             B['c'] holds the abcissae
             B['d'] holds the embedding weights
             B['p'] holds the method order
             B['q'] holds the embedding order
    """
    gamma = 1 - 1/np.sqrt(2);
    A = np.array(((gamma, 0), (1-2*gamma, gamma)),
                 dtype=float)
    b = np.array((0.5, 0.5), dtype=float)
    d = np.array((5/12, 7/12), dtype=float)
    c = np.array((gamma, 1-gamma), dtype=float)
    p = 2
    q = 1
    B = {'A': A, 'b': b, 'c': c, 'd': d, 'p': p, 'q': q}
    return B

def ESDIRK32():
    """
    Usage: B = ESDIRK32()

    Utility routine to return the an ESDIRK table corresponding to
    a 5-stage, 3rd-order method with 2nd-order embedding.

    Outputs: B['A'] holds the stage coefficients
             B['b'] holds the solution weights
             B['c'] holds the abcissae
             B['d'] holds the embedding weights
             B['p'] holds the method order
             B['q'] holds the embedding order
    """
    A = np.array(((0, 0, 0, 0, 0),
                  (9/40, 9/40, 0, 0, 0),
                  (9*(1+np.sqrt(2))/80, 9*(1+np.sqrt(2))/80, 9/40, 0, 0),
                  ((22+15*np.sqrt(2))/80/(1+np.sqrt(2)), (22+15*np.sqrt(2))/80/(1+np.sqrt(2)), -7/40/(1+np.sqrt(2)), 9/40, 0),
                  ((2398+1205*np.sqrt(2))/2835/(4+3*np.sqrt(2)), (2398+1205*np.sqrt(2))/2835/(4+3*np.sqrt(2)), -2374*(1+2*np.sqrt(2))/2835/(5+3*np.sqrt(2)), 5827/7560, 9/40)), dtype=float)
    b = np.array(((2398+1205*np.sqrt(2))/2835/(4+3*np.sqrt(2)), (2398+1205*np.sqrt(2))/2835/(4+3*np.sqrt(2)), -2374*(1+2*np.sqrt(2))/2835/(5+3*np.sqrt(2)), 5827/7560, 9/40), dtype=float)
    c = np.array((0, 9/20, 9*(2+np.sqrt(2))/40, 3/5, 1), dtype=float)
    d = np.array((4555948517383/24713416420891, 4555948517383/24713416420891, -7107561914881/25547637784726, 30698249/44052120, 49563/233080), dtype=float)
    p = 3
    q = 2
    B = {'A': A, 'b': b, 'c': c, 'd': d, 'p': p, 'q': q}
    return B

def ESDIRK43():
    """
    Usage: B = ESDIRK43()

    Utility routine to return the an ESDIRK table corresponding to
    a 7-stage, 4th-order method with 3rd-order embedding.

    Outputs: B['A'] holds the stage coefficients
             B['b'] holds the solution weights
             B['c'] holds the abcissae
             B['d'] holds the embedding weights
             B['p'] holds the method order
             B['q'] holds the embedding order
    """
    b = np.array((0, -5649241495537/14093099002237, 5718691255176/6089204655961, 2199600963556/4241893152925, 8860614275765/11425531467341, -3696041814078/6641566663007, 1/8), dtype=float)
    b[0] = 1-np.sum(b)
    c = np.array((0, 1/4, 1200237871921/16391473681546, 1/2, 395/567, 89/126, 1), dtype=float)
    d = np.array((0, -1517409284625/6267517876163, 8291371032348/12587291883523, 5328310281212/10646448185159, 5405006853541/7104492075037, -4254786582061/7445269677723, 19/140), dtype=float)
    d[0] = 1-np.sum(d)
    A = np.array(((0, 0, 0, 0, 0, 0, 0),
                  (0, 1/8, 0, 0, 0, 0, 0),
                  (0, -39188347878/1513744654945, 1/8, 0, 0, 0, 0),
                  (0, 1748874742213/5168247530883, -1748874742213/5795261096931, 1/8, 0, 0, 0),
                  (0, -6429340993097/17896796106705, 9711656375562/10370074603625, 1137589605079/3216875020685, 1/8, 0, 0),
                  (0, 405169606099/1734380148729, -264468840649/6105657584947, 118647369377/6233854714037, 683008737625/4934655825458, 1/8, 0), b), dtype=float)
    A[1,0] = c[1]-np.sum(A[1,:])
    A[2,0] = c[2]-np.sum(A[2,:])
    A[3,0] = c[3]-np.sum(A[3,:])
    A[4,0] = c[4]-np.sum(A[4,:])
    A[5,0] = c[5]-np.sum(A[5,:])
    p = 4
    q = 3
    B = {'A': A, 'b': b, 'c': c, 'd': d, 'p': p, 'q': q}
    return B

def ESDIRK54():
    """
    Usage: B = ESDIRK54()

    Utility routine to return the an ESDIRK table corresponding to
    a 7-stage, 5th-order method with 4th-order embedding.

    Outputs: B['A'] holds the stage coefficients
             B['b'] holds the solution weights
             B['c'] holds the abcissae
             B['d'] holds the embedding weights
             B['p'] holds the method order
             B['q'] holds the embedding order
    """
    c = np.array((0,  46/125, 7121331996143/11335814405378,
                  49/353, 3706679970760/5295570149437, 347/382, 1),
                  dtype=float)
    b = np.array((0, -188593204321/4778616380481,
                  2809310203510/10304234040467, 1021729336898/2364210264653,
                  870612361811/2470410392208, -1307970675534/8059683598661,
                  23/125), dtype=float)
    b[0] = 1-np.sum(b)
    d = np.array((0, -582099335757/7214068459310, 615023338567/3362626566945,
                  3192122436311/6174152374399, 6156034052041/14430468657929,
                  -1011318518279/9693750372484, 1914490192573/13754262428401),
                  dtype=float)
    d[0] = 1-np.sum(d)
    A = np.array(((0, 0, 0, 0, 0, 0, 0),
                  (0, 23/125, 0, 0, 0, 0, 0),
                  (0, 791020047304/3561426431547, 23/125, 0, 0, 0, 0),
                  (0, -158159076358/11257294102345,
                   -85517644447/5003708988389, 23/125, 0, 0, 0),
                  (0, -1653327111580/4048416487981,
                   1514767744496/9099671765375, 14283835447591/12247432691556,
                   23/125, 0, 0),
                  (0, -4540011970825/8418487046959,
                   -1790937573418/7393406387169, 10819093665085/7266595846747,
                   4109463131231/7386972500302, 23/125, 0),
                  b), dtype=float)
    A[1,0] = c[1]-np.sum(A[1,:])
    A[2,0] = c[2]-np.sum(A[2,:])
    A[3,0] = c[3]-np.sum(A[3,:])
    A[4,0] = c[4]-np.sum(A[4,:])
    A[5,0] = c[5]-np.sum(A[5,:])
    p = 5
    q = 4
    B = {'A': A, 'b': b, 'c': c, 'd': d, 'p': p, 'q': q}
    return B

def ESDIRK843():
    """
    Usage: B = ESDIRK843()

    Utility routine to return the ESDIRK table corresponding to
    a fourth-order method a semilinear order 3.

    Outputs: B['A'] holds the stage coefficients
             B['b'] holds the solution weights
             B['c'] holds the abcissae
             B['d'] holds the embedding weights
             B['p'] holds the method order
             B['q'] holds the embedding order
    """
    A = np.array(((0, 0, 0, 0, 0, 0, 0, 0),
                  (31/125, 31/125, 0, 0, 0, 0, 0, 0),
                  (3781/15500, 63/124, 31/125, 0, 0, 0, 0, 0),
                  (-3882222210210885/75584786387396543, -3882222210210885/75584786387396543, 0, 31/125, 0, 0, 0, 0),
                  (34038088698073943/122803704925069405, 33514318812866834/119213963756997001, 224887786579749/60595115130919582, 4253927007933940/115874777755193681, 31/125, 0, 0, 0),
                  (137658149652207956/209706981851726679, 126492513018975825/128664872604952432, 11417678526293581/37223122310090063, -103353126478507816/135174297737314759, -5/7, 31/125, 0, 0),
                  (117786594983325079/151727549241844762, 18526475144695067/21268081176298953, 7206985701555927/80976168939093068, -94099054066115167/95522373062038575, -26/27, 190069087194766309/197235632620571833, 31/125, 0),
                  (-29602757552094/1071399797354437, 548139805377293/112676962442277364, 424515922983497/13913811815644881, 67814305287223931/162780150685834614, -41854401642916128/116143966895455495, 74958030483037457/95092867575812394, -21188129/211373000, 31/125)), dtype=float)
    b = A[-1,:]
    c = np.sum(A, axis=1)
    d = A[-2,:]
    p = 4
    q = 3
    B = {'A': A, 'b': b, 'c': c, 'd': d, 'p': p, 'q': q}
    return B

def EDIRK1054():
    """
    Usage: B = EDIRK1054()

    Utility routine to return the EDDIRK table corresponding to
    a 5th-order accurate method with semilinear order 4.

    Outputs: B['A'] holds the stage coefficients
             B['b'] holds the solution weights
             B['c'] holds the abcissae
             B['d'] holds the embedding weights
             B['p'] holds the method order
             B['q'] holds the embedding order
    """
    A = np.array(((0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                  (23704630662296147/85251944606883963, 23704630662296147/85251944606883963, 0, 0, 0, 0, 0, 0, 0, 0),
                  (-3828053375722559/66474452137650668, -3828053375722559/66474452137650668, 23704630662296147/85251944606883963, 0, 0, 0, 0, 0, 0, 0),
                  (-1768624057837184/97907440837578655, -1768624057837184/97907440837578655, 32342625154963567/63657230586675651, 23704630662296147/85251944606883963, 0, 0, 0, 0, 0, 0),
                  (38689688244273643/145529658745107532, 38689688244273643/145529658745107532, 5165440406558499/48318409929685292, 0, 23704630662296147/85251944606883963, 0, 0, 0, 0, 0),
                  (-2048929420167937/62953617727398324, -2048929420167937/62953617727398324, -5242126029351595/74777062100226619, 0, 0, 23704630662296147/85251944606883963, 0, 0, 0, 0),
                  (-14097385432048041/83960854026807536, -14097385432048041/83960854026807536, 102631915790147645/94805799854610222, -5088592904909032/78095945223394903, 4194495314217601/126041470655949480, -25019099907264765/59359248890957054, 23704630662296147/85251944606883963, 0, 0, 0),
                  (-6666023823723632/60782333039950069, -6666023823723632/60782333039950069, 34950019030688054/58700988111542175, 16157137791883982/129136549150431411, -4633364877709012/111280515475374397, -44830020893037844/115925512623522465, -10683392218257989/83044047426145149, 23704630662296147/85251944606883963, 0, 0),
                  (7974359957524127/41777369871990865, 7974359957524127/41777369871990865, 31122288307425661/48758046711807126, 124912727797607611/63031353902083464, -68608793789563332/113404568873370149, -95359235367355842/59441684582698261, -162051494287025479/83831556722521602, 148921337658127561/79960515909413127, 23704630662296147/85251944606883963, 0),
                  (-10206283873289495/82173081853978556, -10206283873289495/82173081853978556, 0, -149692166756442484/122226586801197919, 59118216399459218/50501318642781983, 239399454367668061/196562057586860935, 62112483136249447/41672907966740429, -117867017378953048/115349226975194417, -40768154109170907/61569993216212962, 23704630662296147/85251944606883963)), dtype=float)
    b = A[-1,:]
    c = np.sum(A, axis=1)
    d = np.array((-7377933185266438/48541560297323275, -7377933185266438/48541560297323275, 0, 39662348139301097/87268009515385808, 68262268363872686/135528001467088899, 58007573143164457/132805403176109435, -28930654619832593/286820031950471742, 15187635870586502/50762060951677207, -247981587118689503/472717409267306743, 4/17), dtype=float)
    p = 5
    q = 4
    B = {'A': A, 'b': b, 'c': c, 'd': d, 'p': p, 'q': q}
    return B

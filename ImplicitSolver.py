# ImplicitSolver.py
#
# Module containing the ImplicitSolver class, to be used within implicit time stepping methods.
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC


class ImplicitSolver:
    """
    The two required arguments when constructing an ImplicitSolver object are a function
    for the Jacobian of the ODE right-hand side function, f_y, and a key denoting the type
    of solver to construct:
        'dense'  = the function f_y(t,y,args) produces a 2D numpy nd-array of Jacobian entries
                   (a dense LU-factorization-based linear solver is used).
        'sparse' = the function f_y(t,y,args) produces a scipy.linalg.sparse matrix of Jacobian
                   entries (a sparse LU-factorization-based linear solver is used).

    Other optional inputs focus on specific Newton-Raphson solver options:
        maxiter  = maximum allowed number of nonlinear iterations (integer >0)
        rtol     = relative solution tolerance (float, >= 1e-15)
        atol     = absolute solution tolerance (float or numpy array with n entries, all >=0)
        Jfreq    = frequency for reconstructing Jacobian solver (i.e., f_y is called every
                   Jfreq iterations)
    """
    def __init__(self, f_y, solver_type, p_setup=0, maxiter=10, rtol=1e-3, atol=0.0, Jfreq=1):

        # required inputs
        self.f_y = f_y
        self.solver_type = solver_type

        # optional inputs
        self.maxiter = maxiter
        self.rtol = rtol
        self.atol = atol
        self.Jfreq = 1
        if (Jfreq > 0):
            self.Jfreq = Jfreq

        # internal data
        self.linear_solver = None
        self.total_iters = 0
        self.total_setups = 0
        if ((solver_type != 'dense') and (solver_type != 'sparse')):
            raise ValueError("Illegal solver_type input, must be one of dense/sparse")

    def setup_linear_solver(self, t, gamma, args=()):
        """
        Creates a function that users can call prior to solve(), to construct
        scipy.sparse.linalg.LinearOperator objects as needed for the Newton-Raphson
        method.  This is designed to be called by the implicit time integrator at
        each implicit step and/or stage.

        args is used for user-provided optional parameters of their Jacobian, f_y.
        """
        import numpy as np
        from scipy.linalg import lu_factor
        from scipy.linalg import lu_solve
        from scipy.sparse import identity
        from scipy.sparse.linalg import LinearOperator
        from scipy.sparse.linalg import factorized

        if (self.solver_type == 'dense'):
            def J(y):
                Jac = np.eye(y.size) + gamma*self.f_y(t,y,*args)
                try:
                    lu, piv = lu_factor(Jac)
                except:
                    raise RuntimeError("Dense Jacobian factorization failure")
                Jsolve = lambda b: lu_solve((lu, piv), b)
                return LinearOperator((y.size,y.size), matvec=Jsolve)
        elif (self.solver_type == 'sparse'):
            def J(y):
                Jac = identity(y.size) + gamma*self.f_y(t,y,*args)
                try:
                    Jfactored = factorized(Jac)
                except:
                    raise RuntimeError("Sparse Jacobian factorization failure")
                Jsolve = lambda b: Jfactored(b)
                return LinearOperator((y.size,y.size), matvec=Jsolve)
        self.linear_solver = J

    def solve(self, Ffcn, y0):
        """
        Implements a modified Newton-Raphson method for approximating a root of the
        nonlinear system of equations F(y)=0.  Here y is a numpy array with n entries,
        and F(y) is a function that outputs an array with n entries.  The iteration
        ceases when

           || (ynew - yold) / (atol + rtol*|ynew|) ||_RMS < 1

        Required inputs:
            Ffcn  = nonlinear residual function, F(y0) should return a numpy array n entries
            y0 = initial guess at solution (numpy array with n entries)

        Output:
            y -- the approximate solution (numpy array with n entries)
            iters -- the number of iterations performed
            success -- True if iteration converged; False otherwise
        """
        import numpy as np

        # ensure that linear_solver has been created
        assert self.linear_solver != None, "linear_solver has not been created"

        # initialize outputs
        y = np.copy(y0)
        iters = 0
        success = False

        # store nonlinear system size
        n = y0.size

        # evaluate initial residual
        F = Ffcn(y)

        # set up initial Jacobian solver
        Jsolver = self.linear_solver(y)
        self.total_setups += 1

        # perform iteration
        for its in range(1,self.maxiter+1):

            # increment iteration counter
            iters += 1
            self.total_iters += 1

            # solve Newton linear system
            h = Jsolver.matvec(F)

            # compute Newton update
            y -= h

            # check for convergence
            if (np.linalg.norm(h / (self.atol + self.rtol*np.abs(y)))/np.sqrt(n) < 1):
                success = True
                return y, iters, success

            # update nonlinear residual
            F = Ffcn(y)

            # update Jacobian every "Jfreq" iterations
            if (its % self.Jfreq == 0):
                Jsolver = self.linear_solver(y)
                self.total_setups += 1

        # if we made it here then iteration did not converge; return with current
        # solution (note that success is still False)
        return y, iters, success

    def get_total_iters(self):
        """ Returns the total number of nonlinear solver iterations over the life of the solver """
        return self.total_iters

    def get_total_setups(self):
        """ Returns the total number of linear solver setup calls over the life of the solver """
        return self.total_setups

    def reset(self):
        """ Resets the solver statistics """
        self.total_iters = 0
        self.total_setups = 0

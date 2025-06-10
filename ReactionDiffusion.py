#!/usr/bin/env python3
#
# This file defines the RHS and Jacobian functions associated with the 
# reaction-diffusion problem,
#
#    u_t = u_{xx} + 1/(1+u^2) + Phi(x,t),  (x,t) in [0,1]^2
#    u(t,0) = u(t,1) = 0,  t in [0,1]
#    u(0,x) = x(1-x)
# that has analytical solution u(x,t) = x(1-x)e^t.  For this problem,
#    Phi(x,t) = u_t - u_{xx} - 1/(1+u^2)
#             = x(1-x)e^t + 2e^t - 1/(1+u^2)
#             = u + 2e^t - 1/(1+u^2)
#
# Note that the Jacobian of this RHS equals 
#    D - diag(2u/(1+u^2)^2)
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
import scipy.sparse as sp

# problem parameters
t0 = 0.0
tf = 1.0
xl = 0.0
xr = 1.0
Nx = 201
dx = (xr-xl)/(Nx-1)
xgrid = np.linspace(xl+dx, xr-dx, Nx-2)

# diffusion matrix (in sparse format)
diags = [-2*np.ones((Nx-2,),dtype=float)/(dx**2), 
         np.ones((Nx-3,),dtype=float)/(dx**2), 
         np.ones((Nx-3,),dtype=float)/(dx**2)]
D = sp.diags_array(diags, offsets=[0,1,-1])

# solution and relevant functions
def utrue(x,t):
    """
    True solution to problem.
    """
    return x*(1-x)*np.exp(t)

def Phi(x,t):
    """
    Forcing function for the IVP.
    """
    return utrue(x,t) + 2*np.exp(t) - 1/(1+utrue(x,t)**2)

def f(t,u):
    """
    RHS function for the IVP.
    """
    return (D @ u + 1/(1+u**2) + Phi(xgrid,t))

def J(t,u):
    """
    Jacobian (in sparse matrix format) of the right-hand side
    function, J(t,y) = df/dy, for the IVP.
    """
    return (D - sp.diags(2*u/((1+u**2)**2)))

def u0():
    """
    Initial condition
    """
    return utrue(xgrid,t0)

# end of file
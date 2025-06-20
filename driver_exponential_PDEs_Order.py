#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 08:54:14 2025

@author: bom
"""
#We consider the following PDE:
#   U_t = U_xx + 1/(1+U^2) + phi(t,x) over [0,1]x[0,1] subject to Dirichlet
# homogenous boundary condition. 
# phi(t,x) is choosen s.t. the exact solution is U(t,x) = x(1-x)*exp(t)

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import scipy.linalg
from kiops import kiops
from integrators import *
import time

# Space interval
a = 0
b = 1
t0 = 0
t_end = 1
m = 100
delta_x = (float(b) - float(a)) / float(m)
x = np.linspace(a, b, m + 1)
X = x[1:-1]

# Initial vector (length m-1)
U0 = X - X**2  

#Define the second derivative operator
e = np.ones(m - 1)
A_matrix = sp.diags([e/delta_x**2, -2*e/delta_x**2, e/delta_x**2], [-1, 0, 1], (m-1, m-1), format='csc')

#Rewrite the problem in autonomous form
A = sp.block_diag(([[0]], A_matrix)).tocsc()
U0 = np.concatenate([[t0], U0])  # prepend t0 to U0

#Define the nonlinear part
def g(U):
    # U[0] = t, U[1:] = spatial part
    t = U[0]
    u_x = U[1:]
    part1 = (2 + X - X**2) * np.exp(t)
    part2 = 1 / (1 + (X * (1 - X) * np.exp(t))**2)
    part3 = 1 / (1 + u_x**2)
    g_val = np.concatenate(([1], part1 - part2 + part3))
    return g_val
#Define the RHS 
def F(U):
    return A @ U + g(U)
#Define the Jacobian
def J(U, X):
    
    m = len(U)
    t = U[0]
    u_x = U[1:]
    # Diagonal block (m-1, m-1)
    K = sp.diags(-2 * u_x / (1 + u_x**2)**2,format='lil')
    # Block diagonal: top left 0, lower block K, result is (m, m)
    Jacobian = sp.block_diag(([[0]], K), format='lil')
    # Assign to (1:, 0)
    extra = (2 + X - X**2) * np.exp(t) + 2 * (X * (1 - X) * np.exp(t))**2 / (1 + (X * (1 - X) * np.exp(t))**2)**2
    Jacobian[1:, 0] = extra  
    return Jacobian.tocsc()  # Convert to CSC for computations

def u_true(x, t):
    return (x - x**2) * np.exp(t)

#Convergence plot
NTS = np.array([2,4,8,16,32])*5
Err_ExpRK2 = np.zeros(len(NTS))
Err_ExpRK3 = np.zeros(len(NTS))
Err_ExpRK4s5 = np.zeros(len(NTS))
Err_ExpRB2 = np.zeros(len(NTS))
Err_ExpRB3 = np.zeros(len(NTS))
Err_ExpRB4s2 = np.zeros(len(NTS))
CPU_time_ExpRK2 = np.zeros(len(NTS))
CPU_time_ExpRK3 = np.zeros(len(NTS))
CPU_time_ExpRB2 = np.zeros(len(NTS))
CPU_time_ExpRB3 = np.zeros(len(NTS))
CPU_time_ExpRB4s2 = np.zeros(len(NTS))
CPU_time_ExpRK4s5 = np.zeros(len(NTS))

for i in range(len(NTS)):
    start = time.perf_counter()
    _, sol_ExpRK2 = expRK2s2a(F, A, g, t0, t_end, U0, NTS[i])
    end = time.perf_counter()
    CPU_time_ExpRK2[i] = end - start
    
    start = time.perf_counter()
    _, sol_ExpRK3 = expRK3s3a(F, A, g, t0, t_end, U0, NTS[i])
    end = time.perf_counter()
    CPU_time_ExpRK3[i] = end - start
    
    start = time.perf_counter()
    _, sol_ExpRK4s5 = expRK4s5(F, A, g, t0, t_end, U0, NTS[i])
    end = time.perf_counter()
    CPU_time_ExpRK4s5[i] = end - start
    
    start = time.perf_counter()
    _, sol_ExpRB2 = expRB2(F, A, J, g, t0, t_end, U0, NTS[i], X)
    end = time.perf_counter()
    CPU_time_ExpRB2[i] = end - start
    
    start = time.perf_counter()
    _, sol_ExpRB3 = expRB3s3(F, A, J, g, t0, t_end, U0, NTS[i], X,0.5)
    end = time.perf_counter()
    CPU_time_ExpRB3[i] = end - start
    
    start = time.perf_counter()    
    _, sol_ExpRB4s2 = expRB3s3(F, A, J, g, t0, t_end, U0, NTS[i], X,0.75)
    end = time.perf_counter()
    CPU_time_ExpRB4s2[i] = end - start
    
    Err_ExpRK2[i] = (np.linalg.norm(sol_ExpRK2[1:] - u_true(X,1)))
    Err_ExpRK3[i] = (np.linalg.norm(sol_ExpRK3[1:] - u_true(X,1)))
    Err_ExpRK4s5[i] = (np.linalg.norm(sol_ExpRK4s5[1:] - u_true(X,1)))
    Err_ExpRB2[i] = (np.linalg.norm(sol_ExpRB2[1:] - u_true(X,1)))
    Err_ExpRB3[i] = (np.linalg.norm(sol_ExpRB3[1:] - u_true(X,1)))
    Err_ExpRB4s2[i] = (np.linalg.norm(sol_ExpRB4s2[1:] - u_true(X,1)))


#Order convergent plot
plt.plot(np.log10(1/NTS),np.log10(Err_ExpRK2),'gp-')
plt.plot(np.log10(1/NTS),np.log10(Err_ExpRK3),'bx-')
plt.plot(np.log10(1/NTS),np.log10(Err_ExpRK4s5),'yx-')
plt.plot(np.log10(1/NTS),np.log10(Err_ExpRB2),'rp-')
plt.plot(np.log10(1/NTS),np.log10(Err_ExpRB3),'bp-')
plt.plot(np.log10(1/NTS),np.log10(Err_ExpRB4s2),'yp-')
plt.plot(np.log10(1/NTS),2*np.log10(1/NTS*5),'rx--')
plt.plot(np.log10(1/NTS),3*np.log10(1/NTS*2),'rx--')
plt.plot(np.log10(1/NTS),4*np.log10(1/NTS),'rx--')
plt.legend(["ExpRK2", "ExpRK3", "ExpRK4s5", "ExpRB2","ExpRB3","ExpRB4", "Slope 2", "Slope 3", "Slope 4"])
plt.xlabel("Log($dt$)")
plt.ylabel("Log(Error)")
plt.title("Convergence order")
plt.savefig('order_conv_plot.png', dpi=300)  # Save as PNG, high quality
plt.show()
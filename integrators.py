#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 14:29:09 2025

@author: bom
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from kiops import kiops
import scipy.linalg

def expRK1(F, A, g, t0, t_end, u0, N):
    dt = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    n = len(u0)
    zero = np.zeros((1, n))
    u = u0.copy()  # To avoid modifying the input

    for i in range(N):
        if t[i] + dt > t_end:
            dt = t_end - t  # Adjust final step
        # Compute Un2
        Fu = F(u)
        input_mat_1 = np.vstack((zero, Fu.flatten()))
        incr1, _ = kiops(np.array([dt]), A, input_mat_1)
        u = u.flatten() + incr1.flatten()
        
    expRK1_sol = u
    return t, expRK1_sol

def expRK2s2a(F, A, g, t0, t_end, u0, N):
    dt = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    n = len(u0)
    zero = np.zeros((1, n))
    u = u0.copy()  # To avoid modifying the input

    for i in range(N):
        if t[i] + dt > t_end:
            dt = t_end - t  # Adjust final step
        # Compute Un2
        Fu = F(u)
        input_mat_1 = np.vstack((zero, Fu.flatten()))
        incr1, _ = kiops(np.array([dt]), A, input_mat_1)
        Un2 = u.flatten() + incr1.flatten()
        # Compute Dn2
        Dn2 = g(Un2) - g(u)
        input_mat_2 = np.vstack((zero, zero, Dn2.flatten()/dt))
        incr2, _ = kiops(np.array([dt]), A, input_mat_2)
        # Update u
        u = Un2 + incr2.flatten()
        
    expRK2_sol = u
    return t, expRK2_sol

def expRK3s3a(F, A, g, t0, t_end, u0, N):
    dt = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    n = len(u0)
    zero = np.zeros((1, n))
    u = u0.copy()  # To avoid modifying the input

    for i in range(N):
        if t[i] + dt > t_end:
            dt = t_end - t  # Adjust final step
        # Compute Un2
        Fu = F(u)
        #Compute Dn2
        input_mat_1 = np.vstack((zero, Fu.flatten()))
        incr1, _ = kiops(dt*np.array([1/3]), A, input_mat_1)
        Un2 = u.flatten() + incr1.flatten()
        Dn2 = g(Un2) - g(u)
        
        #Compute Dn3
        input_mat_2 = np.vstack((zero, Fu.flatten(),Dn2.flatten()/dt*3))
        incr2, _ = kiops(dt*np.array([2/3]), A, input_mat_2)
        Un3 = u.flatten() + incr2.flatten()
        Dn3 = g(Un3) - g(u)
        # Update u
        input_mat_3 = np.vstack((zero, Fu.flatten(),Dn3.flatten()/dt*3/2))
        incr3, _ = kiops(dt*np.array([1]), A, input_mat_3)
        u = u.flatten() + incr3.flatten()

    expRK3s3_sol = u
    return t, expRK3s3_sol

def expRK4s5(F, A, g, t0, t_end, u0, N):
    dt = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    n = len(u0)
    zero = np.zeros((1, n))
    u = u0.copy()  # To avoid modifying the input

    for i in range(N):
        if t[i] + dt > t_end:
            dt = t_end - t  # Adjust final step
        # Compute Un2
        Fu = F(u)
        #Compute Dn2
        input_mat_1 = np.vstack((zero, Fu.flatten()))
        incr1, _ = kiops(np.array([1]), 0.5*dt*A, input_mat_1)
        Un2 = u.flatten() + 0.5*dt*incr1.flatten()
        Dn2 = g(Un2) - g(u)
        
        #Compute Dn3
        input_mat_2 = np.vstack((zero, 0.5*Fu.flatten(),Dn2.flatten()))
        incr2, _ = kiops(np.array([1]), 0.5*dt*A, input_mat_2)
        Un3 = u.flatten() + dt*incr2.flatten()
        Dn3 = g(Un3) - g(u)
        
        #Compute Dn4
        input_mat_3 = np.vstack((zero, Fu.flatten(),(Dn2+Dn3).flatten()))
        incr3, _ = kiops(np.array([1]), dt*A, input_mat_3)
        Un4 = u.flatten() + dt*incr3.flatten()
        Dn4 = g(Un4) - g(u)
        
        #Compute Dn5
        input_mat_4 = np.vstack((zero, 0.5*Fu.flatten(),0.25*(2*Dn2+2*Dn3-Dn4).flatten(),0.5*(-Dn2-Dn3+Dn4).flatten()))
        incr4, _ = kiops(np.array([1]), 0.5*dt*A, input_mat_4)
        input_mat_5 = np.vstack((zero, zero,0.25*(Dn2+Dn3-Dn4).flatten(),(-Dn2-Dn3+Dn4).flatten()))
        incr5, _ = kiops(np.array([1]), dt*A, input_mat_5)
        Un5 = u.flatten() + dt*incr4.flatten() + dt*incr5.flatten()
        Dn5 = g(Un5) - g(u)
        
        # Update u
        input_mat_6 = np.vstack((zero, Fu.flatten(),(-Dn4+4*Dn5).flatten(),(4*Dn4-8*Dn5).flatten()))
        incr6, _ = kiops(np.array([1]), dt*A, input_mat_6)
        u = u.flatten() + dt*incr6.flatten()

    expRK4s5_sol = u
    return t, expRK4s5_sol

def expRB2(F, A, J, g, t0, t_end, u0, N, X):
    dt = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    n = len(u0)
    zero = np.zeros((1, n))
    u = u0.copy()  # To avoid modifying the input

    for i in range(N):
        if t[i] + dt > t_end:
            dt = t_end - t  # Adjust final step
        # Update u
        Fu = F(u)
        Jn = A+J(u,X)
        input_mat_1 = np.vstack((zero, Fu.flatten()))
        incr1, _ = kiops(np.array([dt]), Jn, input_mat_1)
        u = u.flatten() + incr1.flatten()
    expRB2_sol = u
    return t, expRB2_sol

def expRB3s3(F, A, J, g, t0, t_end, u0, N, X,c2):
    dt = (t_end - t0) / N
    t = np.linspace(t0, t_end, N + 1)
    n = len(u0)
    zero = np.zeros((1, n))
    u = u0.copy()  # To avoid modifying the input

    for i in range(N):
        if t[i] + dt > t_end:
            dt = t_end - t  # Adjust final step
        # Compute Un2
        Fu = F(u)
        Jn = A+J(u,X)
        input_mat_1 = np.vstack((zero, Fu.flatten()))
        incr1, _ = kiops(np.array([c2*dt]), Jn, input_mat_1)
        Un2 = u.flatten() + incr1.flatten()
        #gn(Un2)-gn(u)
        Dn2 = F(Un2) - Jn @ Un2 - F(u) + Jn @ u
        #Update u
        input_mat_2 = np.vstack((zero, Fu.flatten(), zero , Dn2*(2/c2**2)/dt**2))
        incr2, _ = kiops(np.array([dt]), Jn, input_mat_2)
        
        u = u.flatten() + incr2.flatten()
    expRB3s3_sol = u
    return t, expRB3s3_sol

def RK2(f, t0, t_end, y0, NTS):
    t_values = [t0]
    t = t0
    y = y0
    h = (t_end-t0)/NTS
    while t < t_end:
        if t + h > t_end:
            h = t_end - t  # Adjust final step
        k1 = f(y)
        k2 = f(y + h * k1)
        y = y + h * 0.5 * (k1 + k2)
        t += h
        t_values.append(t)
    return np.array(t_values), y

def RK4(f, t0, t_end, y0, NTS):
    t_values = [t0]
    t = t0
    y = y0
    h = (t_end-t0)/NTS
    while t < t_end:
        if t + h > t_end:
            h = t_end - t  # Adjust final step
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        t_values.append(t)
    return np.array(t_values), y


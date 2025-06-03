#!/usr/bin/env python3
#
# Script to solve the IVP system
#    u1'' = u1 + 2*u2' - muh*(u1+mu)/D1 - mu*(u1-muh)/D2,
#    u2'' = u2 - 2*u1' - muh*u2/D1 - mu*u2/D2,
# with initial conditions
#    u1(0) = 0.994,  u2(0) = 0,  u1'(0) = 0,
#    u2'(0) = -2.00158510637908252240537862224
# over the time interval [0,17.1]
#
# Let y0 = u1,  y1 = u1',  y2 = u2,  y3 = y2', then this is equivalent to
#    y0' = y1,
#    y1' = y0 + 2*y3 - muh*(y0+mu)/D1 - mu*(y0-muh)/D2,
#    y2' = y3,
#    y3' = y2 - 2*y1 - muh*y2/D1 - mu*y2/D2,
# with initial conditions
#    y0(0) = 0.994,  y1(0) = 0,  y2(0) = 0,
#    y3(0) = -2.00158510637908252240537862224
# over the time interval [0,17.1]
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

import numpy as np
import matplotlib.pyplot as plt
from AdaptERK import *
from ERK import *

# problem time interval
t0 = 0.0
tf = 17.1

# initial condition
y0 = np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224])

# problem-definining function
def f(t,y):
    """ ODE RHS function """
    mu = 0.012277471
    muh = 1.0-mu
    D1 = ((y[0]+mu)**2 + y[2]**2)**1.5
    D2 = ((y[0]-muh)**2 + y[2]**2)**1.5
    return np.array([y[1],
                     y[0] + 2.0*y[3] - muh*(y[0]+mu)/D1 - mu*(y[0]-muh)/D2,
                     y[3],
                     y[2] - 2.0*y[1] - muh*y[2]/D1 - mu*y[2]/D2])

# shared testing data
Nout = 101   # includes initial condition
tspan = np.linspace(t0, tf, Nout)

# tolerances
rtol = 1.e-6
atol = 1.e-12

# create adaptive Dormand--Prince, adaptive Bogacki-Shampine, and ERK4 steppers
BS = AdaptERK(f, y0, BogackiShampine(), rtol=rtol, atol=atol, save_step_hist=True)
DP = AdaptERK(f, y0, DormandPrince(), rtol=rtol, atol=atol, save_step_hist=True)
E4 = ERK(f, ERK4())

######## adaptive runs ########
print("\nAdaptive Bogacki-Shampine solver:")
Y_BS, success = BS.Evolve(tspan, y0)
step_hist_BS = BS.get_step_history()
print("  steps = %5i  fails = %2i\n" % (BS.get_num_steps(), BS.get_num_error_failures()))

print("\nAdaptive Dormand-Prince solver:")
Y_DP, success = DP.Evolve(tspan, y0)
step_hist_DP = DP.get_step_history()
print("  steps = %5i  fails = %2i\n" % (DP.get_num_steps(), DP.get_num_error_failures()))

######## fixed-step runs ########
print("\nERK4 runs:")
print("  100 steps:")
Y_erk4_100, success = E4.Evolve(tspan, y0, h=tf/100)

print("  1000 steps:")
Y_erk4_1000, success = E4.Evolve(tspan, y0, h=tf/1000)

print("  10000 steps:")
Y_erk4_10000, success = E4.Evolve(tspan, y0, h=tf/10000)

print("  20000 steps:")
Y_erk4_20000, success = E4.Evolve(tspan, y0, h=tf/20000)

# create plots for adaptive runs
plt.figure()
plt.plot(Y_BS[:,0],Y_BS[:,2])
plt.xlabel('$u_1$')
plt.ylabel('$u_2$')
plt.title('Orbit (reference)')
plt.savefig('adaptive_BS_orbit.png')
plt.figure()
plt.plot(Y_DP[:,0],Y_DP[:,2])
plt.xlabel('$u_1$')
plt.ylabel('$u_2$')
plt.title('Orbit (reference)')
plt.savefig('adaptive_DP_orbit.png')
plt.figure()
plt.plot(step_hist_BS['t'], step_hist_BS['h'], 'b-', label='Bogacki-Shampine')
plt.plot(step_hist_DP['t'], step_hist_DP['h'], 'r-', label='Dormand-Prince')
for i in range(len(step_hist_BS['t'])):
    if (step_hist_BS['err'][i] > 1.0):
        plt.plot(step_hist_BS['t'][i], step_hist_BS['h'][i], 'bx')
for i in range(len(step_hist_DP['t'])):
    if (step_hist_DP['err'][i] > 1.0):
        plt.plot(step_hist_DP['t'][i], step_hist_DP['h'][i], 'rx')
plt.xlabel('$t$')
plt.ylabel('$h$')
plt.title('Adaptive step history')
plt.legend()
plt.savefig('adaptive_ERK_steps.png')

# create plots for fixed-step runs
plt.figure()
plt.plot(Y_erk4_100[:,0],Y_erk4_100[:,2])
plt.xlabel('$u_1$')
plt.ylabel('$u_2$')
plt.title('Orbit (100 steps)')
plt.savefig('orbit_100.png')
plt.figure()
plt.plot(Y_erk4_1000[:,0],Y_erk4_1000[:,2])
plt.xlabel('$u_1$')
plt.ylabel('$u_2$')
plt.title('Orbit (1000 steps)')
plt.savefig('orbit_1000.png')
plt.figure()
plt.plot(Y_erk4_10000[:,0],Y_erk4_10000[:,2])
plt.xlabel('$u_1$')
plt.ylabel('$u_2$')
plt.title('Orbit (10000 steps)')
plt.savefig('orbit_10000.png')
plt.figure()
plt.plot(Y_erk4_20000[:,0],Y_erk4_20000[:,2])
plt.xlabel('$u_1$')
plt.ylabel('$u_2$')
plt.title('Orbit (20000 steps)')
plt.savefig('orbit_20000.png')

# output observations
print("\n  The first fixed step size result that is at all qualitatively correct\n  is the 10000-step run; however the orbits are 'tilted' a bit, and\n  these don't properly line up until the 20000-step run.  In the  adaptive,\n  10000-step and 20000-step runs, the plot is a bit 'kinky' at the right\n  end, but this is due to the coarse set of output times that we store.\n  It's also shocking that the adaptive run obtains the 'correct' answer\n  using only 402 steps (334 successful, 68 failed), which is a tiny\n  fraction of the work required with the fixed-step runs.")

# display all plots; these can be interacted with using the mouse
plt.show()


# end of script

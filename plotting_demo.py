#!/usr/bin/env python3
#
# Python Plotting Introduction Script
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

# import student code
import numpy as np
import matplotlib.pyplot as plt
import sys

# get problem size from command line, otherwise set to 201
N = 201
if (len(sys.argv) > 1):
    N = int(sys.argv[1])
print("Running plotting demo using vectors of size N = ", N)

# create x data
x = np.linspace(-1.0, 1.0, N)

# create function data (first 5 odd-degree Chebyshev polynomials)
T = np.zeros((N,5))
for j in range(5):
    for i in range(N):
        T[i,j] = np.cos((j*2+1.0) * np.arccos(x[i]))

# plot similarly to Matlab; only "legend" command differs
# note that all text fields may include $$ for LaTeX rendering
plt.figure(1)
plt.plot(x,T)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Chebyshev polynomials')
plt.legend(('$T_1(x)$', '$T_3(x)$', '$T_5(x)$', '$T_7(x)$', '$T_9(x)$'))
plt.savefig('figure1.png')

# we can also plot with manual colors and line styles, using
# identical formatting specifiers as in Matlab
plt.figure(2)
plt.plot(x, T[:,0], 'b-',  label='$T_1(x)$')
plt.plot(x, T[:,1], 'r--', label='$T_3(x)$')
plt.plot(x, T[:,2], 'm:',  label='$T_5(x)$')
plt.plot(x, T[:,3], 'g-.', label='$T_7(x)$')
plt.plot(x, T[:,4], 'c-', label='$T_9(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Chebyshev polynomials')
plt.legend()
plt.savefig('figure2.pdf')

# display all plots; these can be interacted with using the mouse
plt.show()

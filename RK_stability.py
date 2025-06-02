#!/usr/bin/env python3
# Function to generate and plot the linear stability regions for Runge--Kutta
# methods.  Includes a simple "main" that uses this function to plot the
# stability region for forward and backward Euler (when posed as RK methods).
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

# general imports
import numpy as np
import matplotlib.pyplot as plt

def RK_stability(A, b, box, N=1000):
    ''' Usage: X,Y = RK_stability(A, b, box, N)

        Inputs:
          A is a Butcher table matrix
          b is a Butcher table gluing coefficients
          box = [xl, xr, yl, yr] is the bounding box for the sub-region
              of the complex plane in which to perform the test
          N is optional, specifying how many sample sub-region points to use

        Outputs:
          (X, Y) where X is an array of real components of the stability boundary
            and Y is an array of imaginary components of the stability boundary

        We consider the RK stability function
          R(eta) = 1 + eta * dot(b, inv(I-eta*A)*e)

        We sample the values in 'box' within the complex plane, plugging
        each value into |R(eta)|, and plot the contour of this function
        having value 1.'''
    import matplotlib.pyplot as pyplot
    import numpy as np

    # extract the components of the Butcher table
    s = len(b)
    e = np.ones(s)
    I = np.diag(e)

    # set mesh of sample points
    xl = box[0]
    xr = box[1]
    yl = box[2]
    yr = box[3]
    x = np.linspace(xl, xr, N)
    y = np.linspace(yl, yr, N)

    # evaluate |R(eta)| for each eta in the mesh
    R = np.empty((N,N))
    for j in range(N):
        for i in range(N):
            eta = x[i] + y[j]*1j;
            R[j,i] = np.abs(1 + eta*np.dot(b, np.linalg.solve(I - eta*A, e)) )

    # create contour
    eps = np.finfo(float).eps
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')
    plt.grid(True)
    plt.plot([box[0],box[1]],[0,0],'k--')
    plt.plot([0,0],[box[2],box[3]],'k--')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(box[0],box[1])
    ax.set_ylim(box[2],box[3])
    contour_set = pyplot.contourf(x, y, R, levels=(-eps,1.0))

    # extract and return vertices in the contour R = 1
    vertices = contour_set.collections[0].get_paths()[0].vertices
    return vertices[:,0], vertices[:,1]


if __name__ == '__main__':
    ''' Driver that calls RK_stability to plot the stability regions for
        forward and backward Euler.'''

    A = np.zeros((1,1))
    b = np.zeros(1)
    b[0] = 1.0
    box = [-3.0, 1.0, -2.0, 2.0]
    (x,y) = RK_stability(A, b, box, 100)
    plt.title('Forward Euler stability region (shaded = stable)')
    plt.savefig('FE_stability.pdf')

    A = np.zeros((1,1))
    A[0,0] = 1.0
    b = np.zeros(1)
    b[0] = 1.0
    box = [-1.0, 3.0, -2.0, 2.0]
    (x,y) = RK_stability(A, b, box, 100)
    plt.title('Backward Euler stability region (shaded = stable)')
    plt.savefig('BE_stability.pdf')

    plt.show()


# end of script

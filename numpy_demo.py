#!/usr/bin/env python3
#
# Basic numpy usage demo script
#
# Daniel R. Reynolds
# Math @ SMU
# Math & Stat @ UMBC

# module imports
import numpy as np

# create a vector of length 5, and write this to screen
a = np.zeros(5)
print("writing array of zeros: ", a)

# create a vector with specific entries
b = np.array( [0.1, 0.2, 0.3, 0.4, 0.5] )
print("writing array of 0.1, 0.2, 0.3, 0.4, 0.5: ", b)

# create a vector using linspace
c = np.linspace(1.0, 5.0, 5)
print ("writing array of 1, 2, 3, 4, 5: ", c)

# verify that b has size 5
if (b.size != 5):
    print("error: incorrect matrix size")

# edit entries of a, and write each entry to screen
a[0] = 10.0
a[1] = 15.0
a[2] = 20.0
a[3] = 25.0
a[4] = 30.0
print("entries of a, one at a time (should give 10, 15, 20, 25, 30):")
for i in range(a.size):
    print("  ", a[i])

# write the values to file
print("writing/reading this same vector to/from the file 'a_data':")
np.savetxt("a_data", a)

# read vector from file
tol = 2.e-15
read_test1a = np.random.random_sample((3,4))
np.savetxt("tmp.txt", read_test1a)
read_test1b = np.loadtxt("tmp.txt")
read_test1_error = read_test1a - read_test1b
if (np.linalg.norm(read_test1_error,np.inf) < tol):
    print("  save/load test 1 passed")
else:
    print("  save/load test 1 failed, ||error|| = ", np.linalg.norm(read_test1_error,np.inf))
    print("    read_test1a = ", read_test1a)
    print("    read_test1b = ", read_test1b)

# testing copy constructor
B = a.copy()
print("B = a.copy(), should give 10, 15, 20, 25, 30: ", B)

# update one entry of a
print("updating the 5th entry of a to be 31:")
a[4] = 31.0
print("   a = ", a)
print("B should not have changed:")
print("   B = ", B)
a[4] = 30.0  # reset to original

# testing array slice constructor
B2 = a[1:4]
print("B2 = a[1:4] uses array slice constructor (NOTE: the last index is not included):")
print("  ",B2)
# update entries of B2
B2[0] = 4.0
B2[1] = 3.0
B2[2] = 2.0
# copy B2 back into a using submatrix copy
a[1:4] = B2
print("span copy back into a, should have entries 10 4 3 2 30")
print("  ", a)
a[1] = 15.0  # reset to original
a[2] = 20.0
a[3] = 25.0

# Test arithmetic operators
print("Testing vector add, should give 1.1, 2.2, 3.3, 4.4, 5.5")
b += c   # b = b + c
print("  ", b)

print("Testing scalar add, should give 2, 3, 4, 5, 6")
c += 1.0  # c = c + 1
print("  ", c)

print("Testing vector subtract, should be 8, 12, 16, 20, 24")
a -= c  # a = a - c
print("  ", a)

print("Testing scalar subtract, should be 0, 1, 2, 3, 4")
c -= 2.0  # c = c - 2
print("  ", c)

print("Testing vector fill, should all be -1")
b = -1*np.ones(b.size)
print("  ", b)

print("Testing scalar multiply, should be 0, 5, 10, 15, 20")
ashallow = c
a = c.copy()
c *= 5.0  # c = c * 5
print("  ", c)

print("Testing shallow copy, should be 0, 5, 10, 15, 20")
print("  ", ashallow)

print("Testing deep copy, should be 0, 1, 2, 3, 4")
print("  ", a)

print("Testing vector multiply, should be 0, -1, -2, -3, -4")
b *= a   # b = b.*a
print("  ", b)

print("Testing vector divide, should be 0, -2.5, -3.3333, -3.75, -4")
j = c.copy()
b += -1.0
j /= b   # j = j ./ b
b += 1.0
print("  ", j)

print("Testing vector +=, should be 0, 4, 8, 12, 16")
b += c
print("  ", b)

print("Testing scalar +=, should be 1, 6, 11, 16, 21")
c += 1.0
print("  ", c)

print("Testing vector -=, should be 1, 2, 3, 4, 5")
c -= b
print("  ", c)

print("Testing scalar -=, should be -2, -1, 0, 1, 2")
a -= 2.0
print("  ", a)

print("Testing vector *=, should be 0, -4, 0, 12, 32")
a *= b
print("  ", a)

print("Testing scalar *=, should be 2, 4, 6, 8, 10")
c *= 2.0
print("  ", c)

print("Testing vector /=, should be 0, -1, 0, 1.5, 3.2")
j = a.copy()
j /= c
print("  ", j)

print("Testing scalar /=, should be 1, 2, 3, 4, 5")
j = c.copy()
j /= 2.0
print("  ", j)

print("Testing scalar fill, should be 3, 3, 3, 3, 3")
a = 3*np.ones(a.size)
print("  ", a)

print("Testing vector norm, should be 14.8324")
print("  ", np.linalg.norm(c))

print("Testing vector infinity norm, should be 10")
print("  ", np.linalg.norm(c,np.inf))

print("Testing vector one norm, should be 30")
print("  ", np.linalg.norm(c,1))

print("Testing vector min, should be 2")
print("  ", np.min(c))

print("Testing vector max, should be 10")
print("  ", np.max(c))

B = np.zeros((2,5))
B[0,0:5] = c  # B(0,:) = c
B[1,:] = a    # B(1,:) = a
B += 2.0

print("Testing matrix infinity norm, should be 40")
print("  ", np.linalg.norm(B,np.inf))

print("Testing matrix one norm, should be 17")
print("  ", np.linalg.norm(B,1))

print("Testing matrix two norm, should be 21.7821")
print("  ", np.linalg.norm(B,2))

print("Testing matrix min, should be 4")
print("  ", np.min(B))

print("Testing matrix max, should be 12")
print("  ", np.max(B))

print("Testing dot, should be 90")
print("  ", np.dot(a, c))

print("Testing logspace, should be 0.01 0.1 1 10 100")
e = np.logspace(-2.0, 2.0, 5)
print("  ", e)

# create and fill in a 10x5 matrix
Y = np.zeros((10,5))
for i in range(10):
    Y[i,0] = 1.0*i
    Y[i,1] = -5.0 + 1.0*i
    Y[i,2] = 2.0 + 2.0*i
    Y[i,3] = 20.0 - 1.0*i
    Y[i,4] = -20.0 + 1.0*i

# extract columns from matrix
Y0 = Y[:,0]
Y1 = Y[:,1]
Y2 = Y[:,2]
Y3 = Y[:,3]
Y4 = Y[:,4]

print("Testing column extraction, should be all zeros:")
Y4 += Y3
print("  ", Y4)

# check linear sum
d = np.linspace(0.0, 4.0, 5)
print("Testing LinearSum, should be 0.02 1.2 4 23 204:")
g = 1.0*d + 2.0*e
print("  ", g)

# check the pow routine
d = d**2  # d = d.^2
print("Testing power, should be 0 1 4 9 16:")
print("  ", d)
d = np.sqrt(d)
print("Testing sqrt, should be 0 1 2 3 4:")
print("  ", d)

# check the abs routine
Y1 = np.abs(Y1)
print("Testing abs, should be 5 4 3 2 1 0 1 2 3 4:")
print("  ", Y1)

# check the transpose routine
print("Testing transpose:")
Z = np.array([[1, 2, 3], [4, 5, 6]])
print(" Z:\n", Z)
print(" Z':\n", np.transpose(Z))

print("Testing matrix product, should be: 9 -1 9 -8 11 6")
A_ = np.eye(6)
A_[0,3] = 2.0
A_[1,2] = -1.0
A_[2,5] = 1.0
A_[3,5] = -2.0
A_[4,5] = 1.0
xtrue_ = np.linspace(1.0, 6.0, 6)
b_ = A_@xtrue_
print("  ", b_)

print("Testing direct linear solver with dense matri:")
C_ = 100*np.eye(9) + np.random.random_sample((9,9))
z_ = np.logspace(-4.0, 4.0, 9)
b_ = C_@z_
try:
    x_ = np.linalg.solve(C_, b_)
except:
    print("  solve failed")
else:
    print("  solve succeeded")
print("  ||x - xtrue|| = ", np.linalg.norm(x_ - z_, np.inf), "\n\n")

#! /usr/bin/env python3
# -*- using: utf-8 -*-

import numpy as np

# LU Decomposition
# A = [[a11, ac^T], [al, A2]], L = [[1, 0^T], [l, L2]], U = [[u11, u^T], [0, U2]]
# u11 = a11, l = al/u11, u^T=ac^T, L2U2 = A2 - l.u^T

n = 6
A = np.matrix([[1.0 / float(i + j) for j in range(n)] for i in range(1, n+1)])
P = np.identity(n)
A_init = A.copy()

def lu_step(s):
    if s == n:
        return
    u11 = A[s, s]
    pos = s
    # pivoting
    for k in range(s, n):
        if A[k, s] >= u11:
            pos = k
            u11 = A[k, s]
    A[[s, pos], :] = A[[pos, s], :]
    P[[s, pos], :] = P[[pos, s], :]
    print("pivot: ({0}, {1})".format(s, pos))
    # memorize L
    A[(s+1):, s] /= u11
    # next step
    A[(s+1):, (s+1):] -=  A[(s+1):, s].dot(A[s, (s+1):])
    lu_step(s + 1)

lu_step(0)

L = A.copy()
for i in range(n):
    for j in range(i, n):
        if i == j:
            L[i, j] = 1
        else:
            L[i, j] = 0
U = A.copy()
for i in range(n):
    for j in range(i):
        U[i, j] = 0

print("P:\n{0}\nA:\n{1}\nL:\n{2}\nU:\n{3}\ndiff:\n{4}".format(P, A_init, L, U, P.dot(A_init)-L.dot(U)))


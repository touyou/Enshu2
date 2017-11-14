#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return np.exp(x**4) - np.pi

def df(x):
    return 3 * (x**3) * np.exp(x**4)

newton_xy = []

def newton(x0):
    x = x0
    print "newton method"
    cnt = 1
    while True:
        newton_xy.append((x, f(x)))
        nx = x - f(x) / df(x)
        if math.isinf(nx) or math.isnan(nx):
            return x
        print cnt, ":", x, "f(x)=", f(x)
        if np.abs(f(nx)) < 0.00001 or np.abs(nx - x) < 0.00001:
            newton_xy.append((nx, f(nx)))
            return nx
        x = nx
        cnt += 1

dec_newton_xy = []

def dec_newton(x0, b, g):
    x = x0
    print "deceleration newton method"
    cnt = 1
    while True:
        dec_newton_xy.append((x, f(x)))
        j = 0
        while np.abs(f(x - g**(-j) * f(x) / df(x))) > (1 - b * g**(-j)) * np.abs(f(x)):
            j += 1
        nx = x - g**(-j) * f(x) / df(x)
        if math.isnan(nx) or math.isinf(nx):
            return x
        print cnt, ":", x, "f(x)=", f(x)
        if np.abs(f(nx)) < 0.00001 or np.abs(nx - x) < 0.00001:
            dec_newton_xy.append((nx, f(nx)))
            return nx
        x = nx
        cnt += 1

nt_ret = newton(0.6)
dnt_ret = dec_newton(0.6, 0.5, 1.5)

print "newton:", nt_ret, "f(x) =", f(nt_ret)
print "deceleration newton:", dnt_ret, "f(x) =", f(dnt_ret)

plt.plot(np.array([x for (x,y) in newton_xy]), np.array([y for (x,y) in newton_xy]), label='newton', marker='o', linestyle='dashed')
plt.plot(np.array([x for (x, y) in dec_newton_xy]), np.array([y for (x,y) in dec_newton_xy]), label='deceleration newton', marker='o', linestyle='dashed')

plt.title('Newton and Deceleration Newton')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc='upper left')

plt.savefig('kadai3.png')

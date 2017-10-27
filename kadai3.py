#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return np.exp(x**4) - np.pi

def df(x):
    return 3 * (x**3) * np.exp(x**4)

def newton(x0):
    x = x0
    print "newton method"
    cnt = 1
    while True:
        nx = x - f(x) / df(x)
        if math.isnan(nx):
            return x
        print cnt, ":", x, f(x), df(x), nx
        if np.abs(f(nx)) < 0.01 or np.abs(nx - x) < 0.01:
            return nx
        x = nx
        cnt += 1

def dec_newton(x0, b, g):
    x = x0
    print "deceleration newton method"
    cnt = 1
    while True:
        j = 0
        while np.abs(f(x - g**(-j) * f(x) / df(x))) > (1 - b * g**(-j)) * np.abs(f(x)):
            j += 1
        nx = x - g**(-j) * f(x) / df(x)
        if math.isnan(nx):
            return x
        print cnt, ":", x, f(x), df(x), j, nx
        if np.abs(f(nx)) < 0.01 or np.abs(nx - x) < 0.01:
            return nx
        x = nx
        cnt += 1

nt_ret = newton(0.01)
dnt_ret = dec_newton(0.01, 0.2, 1.3)

print "newton:", nt_ret, "f(x) =", f(nt_ret)
print "deceleration newton:", dnt_ret, "f(x) =", f(dnt_ret)
